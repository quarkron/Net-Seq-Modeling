"""
netseq_tasep_fast.py
====================
Numba-accelerated Python port of the sjkimlab NETSEQ TASEP simulation.

RELATIONSHIP TO THE MATLAB CODE
--------------------------------
The MATLAB reference implementation (sjkimlab_NETSEQ_TASEP.m) is a straightforward
1-D stochastic simulation: arrays grow dynamically, the inner loop is interpreted
MATLAB, and the NETseq output is a full (geneLength × simtime) matrix.

This Python file preserves the exact same **biology and algorithm** but makes three
engineering changes that together give ~10-50× speedups:

  1. Numba JIT (`@njit`)  — the entire inner loop is compiled to native machine code
     by Numba.  This removes Python interpreter overhead, which dominates for tight
     per-particle loops over thousands of time steps.

  2. Static preallocation — MATLAB grows rnap_locs / RNAP_exitTimes dynamically.
     Numba cannot work with variable-length Python lists or MATLAB-style cell arrays,
     so all buffers are preallocated up-front to conservatively estimated capacities
     (`active_cap`, `history_cap`).

  3. Snapshot-only NETseq — MATLAB returns a full (geneLength × simtime) matrix and
     NETSEQ_simulations.m then extracts 7 time columns (t=200,400,...,1400).  Here
     `_compute_netseq_sum` does that extraction directly and sums into a single 1-D
     vector of length geneLength.  This avoids allocating ~3 GB for a 3075 × 2000
     float64 matrix per run.

ALGORITHM OVERVIEW (identical to MATLAB)
-----------------------------------------
This is a 1-D TASEP (Totally Asymmetric Simple Exclusion Process):
  - Particles (RNAPs, ribosomes, Rho helicase) move rightward along a 1-D lattice.
  - Steric exclusion: no two particles may overlap.
  - Time advances in discrete steps dt=0.1 s from t=0 to t=simtime=2000 s.

Within each dt, four sequential sub-steps run (same order as MATLAB):
  1. RNAP loading at the promoter (active phase, t < glutime only)
  2. RNAP elongation + collision detection
  3. Rho factor loading + chase + premature termination check
  4. Ribosome elongation + RNAP-ribosome coupling

SENTINEL VALUES (identical to MATLAB)
--------------------------------------
rnap_locs[i] encoding:
  1 … geneLength      active RNAP at that nucleotide position
  geneLength+1        RNAP finished (ran off 3′ end normally)
  geneLength+10       RNAP prematurely terminated by Rho

rho_locs[i] encoding:
  0                   no Rho loaded for this RNAP
  1 … geneLength      Rho is chasing this RNAP at that position
  geneLength+9        Rho is done (RNAP was already terminated)

KEY DATA STRUCTURE DIFFERENCE vs MATLAB
-----------------------------------------
MATLAB:   RNAP_exitTimes(pos, rnap_idx)   — rows=positions, cols=RNAP index
Python:   rnap_exit_matrix[rnap_idx, pos] — rows=RNAP index, cols=positions

The axes are transposed.  The reason: NumPy stores arrays in row-major (C) order,
so iterating over columns of a row is the fast direction.  Reading all positions of
a single RNAP sequentially (`rnap_exit_matrix[idx, :]`) is therefore cache-friendly.
Numba inherits the same memory layout.

RNAP ORDERING (same as MATLAB):
  rnap_locs[0] = first RNAP to load (furthest downstream / closest to 3′ end)
  rnap_locs[n_rnap-1] = most recently loaded RNAP (nearest to promoter)
  For collision: RNAP `idx` checks against `idx-1` (the one *ahead* of it).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


# ── Snapshot time points (seconds) ──────────────────────────────────────────
# These mirror the column indices used in NETSEQ_simulations.m:
#   NETseqSum = sum(temp_NETseq(:, 200:200:1500), 2)
# MATLAB uses integer time indices 1..simtime as column headers.
# We evaluate at t=200,400,...,1400 s (the same 7 snapshots) and sum into one
# 1-D signal, skipping the full time-resolved matrix.
SNAPSHOTS = [200, 400, 600, 800, 1000, 1200, 1400]

# ── Preallocation tuning constants ──────────────────────────────────────────
# ACTIVE_CAP_SAFETY_MARGIN: multiplicative headroom above the theoretical
#   maximum packing (gene_length / RNAP_width RNAPs can fit simultaneously).
#   Set > 1.0 to handle transient over-crowding without crashing Numba.
ACTIVE_CAP_SAFETY_MARGIN = 1.25

# HISTORY_CAP_BUFFER_FACTOR and _ADD: the history array must hold *all* RNAPs
#   that ever loaded, not just the active ones.  Expected total loads ≈
#   k_loading × simtime, so we pad with a factor of 2 plus a fixed addend.
HISTORY_CAP_BUFFER_FACTOR = 2.0
HISTORY_CAP_BUFFER_ADD = 32

# HISTORY_CAP_MIN_GAP: minimum slack between active_cap and history_cap,
#   ensuring there is always room for at least a few completed RNAPs.
HISTORY_CAP_MIN_GAP = 8



# ══════════════════════════════════════════════════════════════════════════════
# _sample_first_passage
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True, nogil=True)
def _sample_first_passage(
    dwell_profile: np.ndarray,
    start_pos: int,
    end_pos: int,
    t_now: float,
    dt: float,
    exit_buffer: np.ndarray,
) -> int:
    """
    The Python/Numba equivalent of MATLAB's core movement idiom:

        tempExitTimes = t + cumsum(exprnd(dwell(pos:pos+window)))
        kept = tempExitTimes(tempExitTimes >= t & tempExitTimes <= t+dt)

    Both implementations answer the same question:
        "How many lattice sites does this particle cross during [t, t+dt]?"

    MATLAB approach
    ---------------
    1. Draw `window` i.i.d. Exp(dwell) samples.
    2. Compute cumulative sum → absolute exit times.
    3. Filter to keep only those in [t, t+dt].
    4. len(kept) = number of bases advanced.

    Current approach (this function)
    --------------------------------
    1. Start with `next_time = t_now`.
    2. For each position from start_pos to end_pos:
         a. Add Exp(dwell[pos]) to next_time.
         b. If next_time > t_now + dt → stop (early exit).
         c. Otherwise, record next_time in exit_buffer[count]; count += 1.
    3. Return count = number of bases advanced.

    The early-exit optimization means we never draw more random numbers than
    we need.  MATLAB draws all `bases_evaluated = ceil(Speed*10*dt)` samples 
    statically, while this optimization dynamically stops once t+dt is crossed.

    Parameters
    ----------
    dwell_profile : array of mean dwell times (s) per position (index 0 unused)
    start_pos     : first position to evaluate (1-indexed, same as MATLAB)
    end_pos       : last position to evaluate (inclusive; clipped to gene end)
    t_now         : current simulation time (seconds)
    dt            : time step width (0.1 s)
    exit_buffer   : pre-allocated scratch array to hold recorded exit times

    Returns
    -------
    count : int — number of positions the particle crossed during [t_now, t_now+dt]
    """
    next_time = t_now
    count = 0
    for pos in range(start_pos, end_pos + 1):
        scale = dwell_profile[pos]
        if scale > 0.0:
            next_time += np.random.exponential(scale)
        # Early exit: this and all subsequent positions would be reached after
        # the current time step ends, so we stop sampling.
        if next_time > t_now + dt:
            break
        exit_buffer[count] = next_time
        count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
# _tasep_core
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True, nogil=True)
def _tasep_core(
    seed: int,
    active_cap: int,
    history_capacity: int,
    gene_length: int,
    simtime: int,
    glutime: float,
    dt: float,
    k_loading: float,
    k_ribo_loading: float,
    pt_percent: float,
    rnap_width: int,
    min_rho_load_rna: int,
    bases_rnap: int,
    bases_ribo: int,
    rho_window: int,
    specific_dwelltime1: np.ndarray,
    ribo_specific_dwelltime1: np.ndarray,
    specific_dwelltime_rho: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Numba-compiled inner simulation loop.

    This function encodes exactly the same algorithm as the main `for t=0:dt:simtime`
    loop in sjkimlab_NETSEQ_TASEP.m.  The key structural changes are:

    - All arrays preallocated (Numba cannot grow Python lists).
    - RNAP state stored in flat 1-D arrays indexed by rnap index, matching
      MATLAB's row-vector style but without dynamic resizing.
    - Exit times stored transposed: `rnap_exit_matrix[rnap_idx, pos]` vs
      MATLAB's `RNAP_exitTimes(pos, rnap_idx)`.
    - Ribosome state is also tracked per-RNAP index, same as MATLAB's
      Ribo_locs(rnap), rho_locs(rnap), RNAP_RiboCoupling(rnap).

    Parameters (all scalars passed explicitly for Numba type inference)
    -------------------------------------------------------------------
    seed              : RNG seed for reproducibility
    active_cap        : max simultaneous active RNAPs (safety check)
    history_capacity  : total rows in rnap_exit_matrix (≥ total ever loaded)
    gene_length       : number of nucleotide positions on the gene
    simtime           : total simulation duration (s)
    glutime           : end of active transcription phase (s)
    dt                : time step (s); MATLAB uses dt=0.1 s, same here
    k_loading         : RNAP promoter initiation rate (1/s)
                        MATLAB: kLoading = 1/20, mean wait = 20 s
    k_ribo_loading    : ribosome loading rate (1/s); 0 = disabled
    pt_percent        : Rho loading scaling factor (KRutLoading, default 0.13)
    rnap_width        : RNAP footprint on DNA (bp); MATLAB: RNAP_width = 35
    min_rho_load_rna  : minimum exposed RNA before Rho can load (nt)
                        MATLAB: minRholoadRNA = 80 - rho_width = 50
    bases_rnap        : look-ahead window for RNAP (bases); MATLAB: ceil(RNAPSpeed*10*dt)
    bases_ribo        : look-ahead window for ribosome; MATLAB: ceil(riboSpeed*10*dt)
    rho_window        : look-ahead window for Rho (bases)
    specific_dwelltime1       : RNAP dwell time per position (s/nt)
                                MATLAB: specificDwelltime1 = avgDwelltime * profile
    ribo_specific_dwelltime1  : ribosome dwell time per position (s/nt), uniform
                                MATLAB: RibospecificDwelltime1 = riboavgDwelltime * ones
    specific_dwelltime_rho    : Rho dwell time per position (s/nt), uniform
                                MATLAB: specificDwelltimeRho = dx/rutSpeed * ones

    Returns
    -------
    rnap_exit_matrix : float64 array shape (history_capacity, gene_length+1)
                       rnap_exit_matrix[i, p] = absolute time RNAP i exited position p
                       (0.0 means not yet reached)
                       Transposed relative to MATLAB's RNAP_exitTimes(pos, rnap).
    n_rnap           : int — total number of RNAPs that loaded during the run
    """
    np.random.seed(seed)

    # ── Sentinel values (identical to MATLAB) ────────────────────────────────
    # term_rnap    : RNAP was prematurely terminated by Rho  (MATLAB: geneLength+10)
    # term_rho     : Rho is done / RNAP already terminated   (MATLAB: geneLength+9)
    # finished_rnap: RNAP ran off the 3′ end normally         (MATLAB: geneLength+1)
    # big_t        : "never" timestamp (> simtime) used to disable future events
    term_rnap = gene_length + 10
    term_rho = gene_length + 9
    finished_rnap = gene_length + 1
    big_t = simtime + 1.0

    # ── Array preallocation ───────────────────────────────────────────────────
    # MATLAB grows these dynamically; Numba requires fixed sizes known at compile time.
    #
    # rnap_exit_matrix[rnap_idx, pos] ↔ MATLAB RNAP_exitTimes(pos, rnap_idx)  [transposed]
    # ribo_exit_matrix[rnap_idx, pos] ↔ MATLAB riboExitTimes(pos, rnap_idx)   [transposed]
    rnap_exit_matrix = np.zeros((history_capacity, gene_length + 1), dtype=np.float64)
    ribo_exit_matrix = np.zeros((history_capacity, gene_length + 1), dtype=np.float64)

    # Position arrays — one entry per RNAP in loading order.
    # rnap_locs[i]         ↔ MATLAB rnap_locs(i)
    # ribo_locs[i]         ↔ MATLAB Ribo_locs(i, 1)
    # rho_locs[i]          ↔ MATLAB rho_locs(i, 1)
    # rnap_ribo_coupling[i]↔ MATLAB RNAP_RiboCoupling(i)
    rnap_locs = np.zeros(history_capacity, dtype=np.int64)
    ribo_locs = np.zeros(history_capacity, dtype=np.int64)
    rho_locs = np.zeros(history_capacity, dtype=np.int64)
    rnap_ribo_coupling = np.zeros(history_capacity, dtype=np.int64)

    # Ribosome loading schedule.  big_t = "never load" (same intent as MATLAB's simtime+1).
    # MATLAB: Riboloadt is per-RNAP; default simtime is used as "never" sentinel.
    ribo_loadt = np.full(history_capacity, big_t, dtype=np.float64)

    # Schedule the first RNAP loading event.
    # MATLAB: loadt = exprnd(1/kLoading)
    # Python:  same draw, with explicit guard for k_loading=0 (MATLAB would divide by 0).
    if k_loading > 0.0:
        loadt = np.random.exponential(1.0 / k_loading)
    else:
        loadt = big_t   # k_loading=0 means no RNAPs ever load

    n_rnap = 0                          # count of RNAPs loaded so far
    n_steps = int(simtime / dt)         # total number of time steps

    # Per-step scratch buffers for _sample_first_passage.
    # MATLAB allocates tempExitTimes inside the loop; here we reuse fixed buffers.
    rnap_exit_buffer = np.empty(bases_rnap + 2, dtype=np.float64)
    ribo_exit_buffer = np.empty(bases_ribo + 2, dtype=np.float64)
    rho_exit_buffer = np.empty(rho_window + 2, dtype=np.float64)

    # ════════════════════════════════════════════════════════════════════════
    # MAIN SIMULATION LOOP
    #
    # Advances time in steps of dt=0.1 s from t=0 to t=simtime.
    #
    # MATLAB: `for t = 0:dt:simtime`
    # Python: integer step index to avoid floating-point accumulation errors.
    #         t_now = step * dt  is exact for a power-of-10 dt.
    #
    # Two simulation phases (same as MATLAB):
    #   Phase 1 — Active transcription (0 ≤ t < glutime = 1600 s):
    #     New RNAPs load at the promoter AND all particles move.
    #   Phase 2 — Runoff (glutime ≤ t ≤ simtime = 2000 s):
    #     No new RNAPs load; existing particles continue to the 3′ end.
    #
    # Four sub-steps per time step (same order as MATLAB sections 1–4):
    #   1. RNAP loading
    #   2. RNAP elongation + collision
    #   3. Rho loading + chase + termination
    #   4. Ribosome elongation + coupling
    # ════════════════════════════════════════════════════════════════════════
    for step in range(n_steps + 1):
        t_now = step * dt

        # ── Sub-step 1: RNAP LOADING ─────────────────────────────────────────
        #
        # Mirrors MATLAB section 1 verbatim.  The stochastic loading schedule is
        # a Poisson process: the next loading event is pre-drawn as loadt ~ Exp(1/kLoading).
        #
        # Case A — Promoter BLOCKED: loading time has arrived but the last RNAP
        #   hasn't cleared enough space.  Condition: (last_rnap_pos - RNAP_width) <= 0
        #   means the most recently loaded RNAP (index n_rnap-1) is still so close
        #   to position 1 that a new 35-bp RNAP cannot fit.
        #   Action: reschedule loadt to a new future time (don't load).
        #   MATLAB: `if(loadt <=t & t<glutime & ~isempty(rnap_locs) & rnap_locs(end)-RNAP_width<=0)`
        if (
            loadt <= t_now
            and t_now < glutime
            and n_rnap > 0
            and (rnap_locs[n_rnap - 1] - rnap_width) <= 0
        ):
            if k_loading > 0.0:
                loadt = t_now + np.random.exponential(1.0 / k_loading)
            else:
                loadt = big_t

        # Case B — Promoter CLEAR: loading time has arrived and there is room.
        #   Condition: either no RNAPs loaded yet, or last RNAP has moved >= RNAP_width.
        #   Action: place new RNAP at position 1 and initialize its state.
        #   MATLAB: `if(loadt<=t & t<glutime & (isempty(rnap_locs)||rnap_locs(end)-RNAP_width>=0))`
        if (
            loadt <= t_now
            and t_now < glutime
            and (n_rnap == 0 or (rnap_locs[n_rnap - 1] - rnap_width) >= 0)
        ):
            # Safety check: Numba cannot grow arrays, so we validate capacity.
            # MATLAB grows rnap_locs automatically; here we error if exceeded.
            active_count = 0
            for j in range(n_rnap):
                if rnap_locs[j] > 0 and rnap_locs[j] <= gene_length:
                    active_count += 1
            if active_count >= active_cap:
                raise ValueError("Active RNAP capacity overflow in _tasep_core.")
            if n_rnap >= history_capacity:
                raise ValueError("RNAP history overflow in _tasep_core.")

            idx = n_rnap
            n_rnap += 1

            # Initialize all state for this new RNAP.
            # MATLAB: rnap_locs(end+1) = 1; RNAP_exitTimes(:,end) = zeros; etc.
            rnap_locs[idx] = 1              # RNAP starts at position 1 (5′ end)
            rnap_ribo_coupling[idx] = 0     # not yet coupled to a ribosome
            ribo_locs[idx] = 0              # ribosome not yet loaded (0 = none)
            rho_locs[idx] = 0              # Rho not yet loaded

            # Schedule next RNAP loading (for the NEXT RNAP, not this one).
            if k_loading > 0.0:
                loadt = t_now + np.random.exponential(1.0 / k_loading)
            else:
                loadt = big_t

            # Schedule ribosome loading for this RNAP's nascent mRNA.
            # MATLAB: Riboloadt(end) = t + exprnd(1/kRiboLoading)
            # NOTE: MATLAB default kRiboLoading=0 would cause a divide-by-zero here.
            # Python guards explicitly with the if/else.
            if k_ribo_loading > 0.0:
                ribo_loadt[idx] = t_now + np.random.exponential(1.0 / k_ribo_loading)
            else:
                ribo_loadt[idx] = big_t   # kRiboLoading=0 → no ribosome ever loads

        # ── Sub-step 2: RNAP ELONGATION ──────────────────────────────────────
        #
        # Mirrors MATLAB section 2: `for rnap = 1:length(rnap_locs)`.
        # RNAPs are processed in loading order (0 = oldest/most downstream).
        #
        # MOVEMENT ALGORITHM (same as MATLAB):
        #   1. Compute look-ahead window end = min(curr_pos + bases_rnap, gene_length).
        #      `bases_rnap = ceil(RNAPSpeed * 10 * dt)` gives enough room even if
        #      the RNAP moves at max speed for the entire dt interval.
        #   2. Call _sample_first_passage → returns how many bases to advance and
        #      fills exit_buffer with absolute exit timestamps.
        #      MATLAB equivalent: cumsum(exprnd(dwell)) + t, then filter by [t, t+dt].
        #   3. Collision check against the RNAP directly ahead (index idx-1).
        #   4. Record exit times and update position.
        for idx in range(n_rnap):
            curr_loc = rnap_locs[idx]
            if curr_loc <= gene_length:
                # Clip look-ahead window to gene end.
                # MATLAB: `if rnap_locs(rnap)+bases_evaluated <= geneLength ... else`
                end_loc = curr_loc + bases_rnap
                if end_loc > gene_length:
                    end_loc = gene_length

                # Sample exit times via the first-passage approach.
                # Returns `advance` = number of bases crossed; exit_buffer holds timestamps.
                advance = _sample_first_passage(
                    specific_dwelltime1,
                    curr_loc,
                    end_loc,
                    t_now,
                    dt,
                    rnap_exit_buffer,
                )

                if advance > 0:
                    allowed_advance = advance

                    # ── COLLISION DETECTION ───────────────────────────────────
                    # Only needed for RNAPs that are not the first/most-downstream one.
                    # MATLAB: `if rnap > 1`
                    if idx > 0:
                        prev_loc = rnap_locs[idx - 1]   # position of RNAP directly ahead

                        # SPECIAL CASE: the RNAP directly ahead was terminated (sentinel).
                        # Search backwards to find the nearest non-terminated RNAP ahead.
                        # MATLAB:
                        #   if PrevRNAPloc == geneLength + 10
                        #     j=1; while rnap_locs(rnap-j)==geneLength+10 && rnap-j>1; j++; end
                        #     if j==rnap || rnap-j<1: PrevRNAPloc=geneLength+1 else: PrevRNAPloc=rnap_locs(rnap-j)
                        if prev_loc == term_rnap:
                            j = 1
                            while (
                                j <= idx
                                and rnap_locs[idx - j] == term_rnap
                                and (idx - j) > 0
                            ):
                                j += 1
                            # All predecessors terminated → treat as no obstacle ahead.
                            if j == idx + 1 or (idx - j) < 0:
                                prev_loc = finished_rnap
                            else:
                                prev_loc = rnap_locs[idx - j]

                        # Overlap formula (same as MATLAB):
                        #   overlap = (curr_pos + advance - 1) - prev_RNAP_pos + RNAP_width
                        # If > 0, the RNAP would intrude into the space occupied by the one ahead.
                        overlap = (curr_loc + advance - 1) - prev_loc + rnap_width

                        # If the predecessor has finished (left the gene), no collision possible.
                        # MATLAB: `if PrevRNAPloc >= geneLength: overlap = 0`
                        if prev_loc >= gene_length:
                            overlap = 0

                        if overlap > 0:
                            # Collision: truncate advance by the overlap amount.
                            # MATLAB: `size(tempRNAP_exitTimes,1) - overlap`
                            allowed_advance = advance - overlap

                    # Record exit times and advance position.
                    # MATLAB: RNAP_exitTimes(curr:curr+advance-1, rnap) = tempRNAP_exitTimes
                    #         rnap_locs(rnap) += advance
                    if allowed_advance > 0:
                        for k in range(allowed_advance):
                            rnap_exit_matrix[idx, curr_loc + k] = rnap_exit_buffer[k]
                        rnap_locs[idx] = curr_loc + allowed_advance

            # ── RIBOSOME LOADING onto this RNAP's mRNA ────────────────────────
            # Load the ribosome once its scheduled time arrives AND the RNAP has
            # exposed at least 30 nt of nascent mRNA (RNAP position ≥ Ribo_width=30).
            # MATLAB: `if Riboloadt(rnap) <= t && rnap_locs(rnap) >= Ribo_width`
            #   After loading, Riboloadt(rnap) = simtime to prevent re-loading.
            # Python: ribo_loadt set to big_t (> simtime) instead.
            if ribo_loadt[idx] <= t_now and rnap_locs[idx] >= 30:
                ribo_loadt[idx] = big_t    # "never load again" — one ribosome per mRNA
                ribo_locs[idx] = 1         # ribosome starts at position 1 of the mRNA
                ribo_exit_matrix[idx, 1] = t_now

        # ── Sub-step 3: RHO FACTOR / PREMATURE TERMINATION ───────────────────
        #
        # Mirrors MATLAB section 3: `for RNA = 1:length(rnap_locs)`.
        # Three actions per RNAP:
        #   (a) Decide whether to load a new Rho factor (PT_Model=2 only here).
        #   (b) Move existing Rho toward the RNAP.
        #   (c) Check if Rho caught the RNAP → terminate.
        #
        # NOTE: Only PT_Model=2 is implemented here (the biologically realistic model).
        # MATLAB also has PT_Model=0 (rate-based at rut sites) and PT_Model=1
        # (simple percentage at rut sites), both omitted in this port.
        for idx in range(n_rnap):

            # ── (a) Rho loading decision (PT_Model=2) ─────────────────────────
            # The exposed (ribosome-unprotected) RNA length behind the RNAP is:
            #   ptrna_size = RNAP_pos - RNAP_width - rho_width(30) - Ribo_pos
            # MATLAB: PTRNAsize = rnap_locs(RNA) - RNAP_width - rho_width - Ribo_locs(RNA)
            #         rho_width=30, RNAP_width=35 → subtracts 35+30=65 and Ribo_pos.
            #
            # Loading probability per time step:
            #   P = (pt_percent * ptrna_size / gene_length) × dt   (but scaled by 100)
            # MATLAB: `100*dt*rand <= PTpercent*PTRNAsize/geneLength`
            # Python: `100.0*dt*np.random.random() <= threshold` — identical test.
            #
            # If Rho loads, it's placed at a random position in the exposed RNA region.
            # The `if temp_rho_loc > rho_locs[idx]` ensures Rho can only advance forward.
            if rnap_locs[idx] < gene_length:
                ptrna_size = rnap_locs[idx] - rnap_width - 30 - ribo_locs[idx]
                if ptrna_size > min_rho_load_rna:
                    threshold = pt_percent * ptrna_size / gene_length
                    if 100.0 * dt * np.random.random() <= threshold:
                        temp_rho_loc = rnap_locs[idx] - rnap_width - int(
                            np.floor(np.random.random() * ptrna_size)
                        )
                        # Rho position can only increase (chases in one direction).
                        if temp_rho_loc > rho_locs[idx]:
                            rho_locs[idx] = temp_rho_loc

            # ── (b) Rho elongation ────────────────────────────────────────────
            # Rho moves along the nascent RNA using the same first-passage approach,
            # but with specificDwelltimeRho (speed = 5 × riboSpeed = 5 × 19 = 95 nt/s).
            # MATLAB: `bases_evaluated = ceil(rutSpeed*dt*10)`
            #         `tempExitTimes = t + cumsum(exprnd(specificDwelltimeRho(...)))`
            rho_loc = rho_locs[idx]
            if rho_loc > 0 and rho_loc < gene_length:
                end_loc = rho_loc + rho_window
                if end_loc > gene_length:
                    end_loc = gene_length
                rho_advance = _sample_first_passage(
                    specific_dwelltime_rho,
                    rho_loc,
                    end_loc,
                    t_now,
                    dt,
                    rho_exit_buffer,
                )
                rho_locs[idx] = rho_loc + rho_advance

            # ── (c) Rho catch check → premature termination ───────────────────
            # If Rho position ≥ RNAP position, Rho has caught the RNAP.
            # Both are set to their "done" sentinels.
            # MATLAB: `if rho_locs(RNA) >= rnap_locs(RNA)`
            #           `rnap_locs(RNA) = geneLength+10; rho_locs(RNA) = geneLength+9;`
            if rho_locs[idx] >= rnap_locs[idx]:
                rnap_locs[idx] = term_rnap
                rho_locs[idx] = term_rho

        # ── Sub-step 4: RIBOSOME ELONGATION ──────────────────────────────────
        #
        # Mirrors MATLAB section 4: `for RNA = 1:size(Ribo_locs,1)`.
        # Handles four ribosome states in priority order (same as MATLAB if/elseif):
        #
        #   State 1 — COUPLED, mid-gene (RNAP_RiboCoupling==1, ribo far from 3′ end):
        #     The ribosome has collided with and is pushing the RNAP.  We synchronize
        #     their exit times: ribosome at position p exits at the same time RNAP
        #     exited position (p + RNAP_width).
        #     MATLAB: riboExitTimes(ribo:geneLength-RNAP_width, RNA) =
        #               RNAP_exitTimes(ribo+RNAP_width:geneLength, RNA)
        #
        #   State 2 — COUPLED, near gene end (ribo > geneLength-30):
        #     Too close to 3′ end for the offset trick.  Ribosome finishes
        #     independently with stochastic exit times.
        #     MATLAB: riboExitTimes(ribo:geneLength,RNA) = t + cumsum(exprnd(dwell(...)))
        #
        #   State 3 — RNAP TERMINATED by Rho (rnap_locs == geneLength+10):
        #     RNAP was prematurely terminated; ribosome translates to the last
        #     position the RNAP reached, then stops.
        #     MATLAB: last position = sum(RNAP_exitTimes(:,RNA) > 0)
        #     Python: scan rnap_exit_matrix[idx, 1:] to find last nonzero entry.
        #
        #   State 4 — FREE ribosome (uncoupled, normal movement):
        #     Same first-passage algorithm as RNAP.  After advance, check for
        #     collision with the RNAP (overlap formula same as RNAP vs RNAP).
        #     If collision → set rnap_ribo_coupling=1 and snap to (RNAP_pos - RNAP_width).
        for idx in range(n_rnap):
            ribo_loc = ribo_locs[idx]
            if ribo_loc > 0 and ribo_loc <= gene_length:

                # ── State 1: Coupled, mid-gene ────────────────────────────────
                # Copy RNAP exit times (shifted by RNAP_width) as ribosome exit times.
                # MATLAB: riboExitTimes(ribo:geneLength-RNAP_width,RNA) =
                #           RNAP_exitTimes(ribo+RNAP_width:geneLength,RNA)
                #         Ribo_locs(RNA) = rnap_locs(RNA) - RNAP_width
                if rnap_ribo_coupling[idx] == 1 and ribo_loc <= gene_length - rnap_width:
                    for pos in range(ribo_loc, gene_length - rnap_width + 1):
                        ribo_exit_matrix[idx, pos] = rnap_exit_matrix[idx, pos + rnap_width]
                    ribo_locs[idx] = rnap_locs[idx] - rnap_width

                # ── State 2: Coupled, near gene end ──────────────────────────
                # `geneLength - 30` threshold matches MATLAB's `geneLength - Ribo_width`.
                # MATLAB: elseif RNAP_RiboCoupling(RNA)==1 && Ribo_locs(RNA)>geneLength-Ribo_width
                elif (
                    rnap_ribo_coupling[idx] == 1
                    and ribo_loc > gene_length - 30
                    and ribo_loc < gene_length + 1
                ):
                    # Ribosome finishes the remaining bases independently.
                    finish_t = t_now
                    for pos in range(ribo_loc, gene_length + 1):
                        dwell = ribo_specific_dwelltime1[pos]
                        if dwell > 0.0:
                            finish_t += np.random.exponential(dwell)
                        ribo_exit_matrix[idx, pos] = finish_t
                    ribo_locs[idx] = finished_rnap

                # ── State 3: RNAP prematurely terminated ──────────────────────
                # Ribosome translates to the last position the RNAP transcribed,
                # then the rest of the mRNA is zeroed (it doesn't exist).
                # MATLAB: last_pos = sum(RNAP_exitTimes(:,RNA) > 0)
                # Python: scan for the last nonzero entry in rnap_exit_matrix[idx, :].
                elif rnap_locs[idx] == term_rnap:
                    # Find the last position the RNAP actually reached.
                    last_idx = 0
                    for pos in range(1, gene_length + 1):
                        if rnap_exit_matrix[idx, pos] > 0.0:
                            last_idx = pos
                    # If RNAP terminated before reaching ribosome, use ribosome's
                    # position as the lower bound.
                    if last_idx < ribo_loc:
                        last_idx = ribo_loc
                    finish_t = t_now
                    for pos in range(ribo_loc, last_idx + 1):
                        dwell = ribo_specific_dwelltime1[pos]
                        if dwell > 0.0:
                            finish_t += np.random.exponential(dwell)
                        ribo_exit_matrix[idx, pos] = finish_t
                    # Zero out positions beyond the truncated transcript.
                    for pos in range(last_idx + 1, gene_length + 1):
                        ribo_exit_matrix[idx, pos] = 0.0
                    ribo_locs[idx] = term_rnap

                # ── State 4: Free ribosome (uncoupled) ───────────────────────
                # Normal independent movement with RNAP collision detection.
                # MATLAB: else block inside ribosome elongation section.
                else:
                    # Sample exit times using first-passage algorithm.
                    end_loc = ribo_loc + bases_ribo
                    if end_loc > gene_length:
                        end_loc = gene_length
                    ribo_advance = _sample_first_passage(
                        ribo_specific_dwelltime1,
                        ribo_loc,
                        end_loc,
                        t_now,
                        dt,
                        ribo_exit_buffer,
                    )

                    # Collision check: ribosome vs RNAP ahead of it.
                    # Same overlap formula as RNAP-RNAP collision.
                    # MATLAB: overlap = (Ribo+advance-1) - rnap_locs(RNA) + RNAP_width
                    overlap = (ribo_loc + ribo_advance - 1) - rnap_locs[idx] + rnap_width

                    # If RNAP has finished (ran off gene end), it's not physically
                    # present, so the ribosome can advance freely past its old position.
                    # MATLAB: `if rnap_locs(RNA)==geneLength+1: overlap=0`
                    if rnap_locs[idx] == finished_rnap:
                        overlap = 0

                    if overlap > 0:
                        # Collision: ribosome caught the RNAP → enable coupling.
                        # MATLAB: `if rnap_locs<=geneLength && boolRNAPRiboCoupling==1`
                        #           `RNAP_RiboCoupling(RNA) = 1`
                        if rnap_locs[idx] <= gene_length:
                            rnap_ribo_coupling[idx] = 1
                        # Truncate advance; snap ribosome to just behind RNAP.
                        allowed_advance = ribo_advance - overlap
                        if allowed_advance > 0:
                            for k in range(allowed_advance):
                                ribo_exit_matrix[idx, ribo_loc + k] = ribo_exit_buffer[k]
                            ribo_locs[idx] = rnap_locs[idx] - rnap_width
                    elif ribo_advance > 0:
                        # No collision: advance freely.
                        for k in range(ribo_advance):
                            ribo_exit_matrix[idx, ribo_loc + k] = ribo_exit_buffer[k]
                        ribo_locs[idx] = ribo_loc + ribo_advance

    return rnap_exit_matrix, n_rnap


# ══════════════════════════════════════════════════════════════════════════════
# _compute_netseq_sum
# ══════════════════════════════════════════════════════════════════════════════
def _compute_netseq_sum(
    rnap_exit_matrix: np.ndarray,
    n_rnap: int,
    gene_length: int,
    snapshots: np.ndarray,
) -> np.ndarray:
    """
    Convert RNAP exit-time matrix into a summed NET-seq signal.

    MATLAB EQUIVALENT (from sjkimlab_NETSEQ_TASEP.m + NETSEQ_simulations.m):
    --------------------------------------------------------------------------
    MATLAB builds the full NETseq matrix during the simulation:

        for t = 1:simtime
            tempNETseq = sum(RNAP_exitTimes(:,:) <= t & RNAP_exitTimes(:,:) > 0, 1);
            tempNETseq = tempNETseq(tempNETseq>0 & tempNETseq<geneLength & max(...) > t);
            tempNETseq = histcounts(tempNETseq, 'BinMethod','integers', ...);
            NETseq(:,t) = tempNETseq;
        end

    Then NETSEQ_simulations.m extracts 7 snapshot columns and sums:

        NETseqSum = sum(temp_NETseq(:, 200:200:1500), 2)

    OPTIMIZATION:
    --------------------
    We skip building the (geneLength × simtime) matrix entirely.  Instead, for
    each snapshot time t_s in SNAPSHOTS, we:

      1. For each RNAP i, count how many positions have exit_time <= t_s and > 0.
         This count equals the RNAP's position at snapshot t_s.
         (NumPy: `temp = np.sum((exits <= t_s) & (exits > 0), axis=1)`)

      2. Mask out RNAPs that:
         - Haven't started yet (count == 0)
         - Have already run off the gene (count >= gene_length)
         - Finished before this snapshot (max exit time ≤ t_s)
         (MATLAB: `tempNETseq(tempNETseq>0 & tempNETseq<geneLength & max(...)>t)`)

      3. Histogram remaining positions into integer bins [1, gene_length].
         (NumPy: `np.bincount`)

      4. Accumulate into netseq_sum (sum over all 7 snapshots, same as MATLAB's `sum(...,2)`).

    Memory: this approach uses O(n_rnap × gene_length) instead of O(gene_length × simtime).
    For a 3075-bp gene and 2000 s simulation: 3075×2000×8 bytes ≈ 49 MB per run, which
    adds up over 300 runs.  The snapshot-only approach avoids this entirely.

    Parameters
    ----------
    rnap_exit_matrix : shape (history_capacity, gene_length+1)
                       rnap_exit_matrix[i, p] = time RNAP i exited position p
    n_rnap           : number of RNAPs that loaded (only rows 0..n_rnap-1 are valid)
    gene_length      : number of nucleotide positions
    snapshots        : 1-D array of snapshot times (seconds)

    Returns
    -------
    netseq_sum : 1-D array of length gene_length (positions 1..gene_length)
                 Sum of RNAP density histograms across all snapshot times.
                 Index 0 (position 0) is sliced off to match the 1-indexed biology.
    """
    netseq_sum = np.zeros(gene_length + 1, dtype=float)
    if n_rnap == 0:
        return netseq_sum[1:]

    # Slice to valid RNAP rows only (rows 0..n_rnap-1).
    # Transposed layout vs MATLAB: exits[i, p] = exit time of RNAP i at position p.
    exits = rnap_exit_matrix[:n_rnap, : gene_length + 1]

    # Precompute the maximum exit time for each RNAP, used to filter out
    # RNAPs that have already completed by a given snapshot.
    # MATLAB: `max(RNAP_exitTimes(:,:), [], 1)` — max over rows for each column.
    # Python: `np.max(..., axis=1)` — max over columns for each row (transposed).
    max_per_rnap = np.max(exits, axis=1)

    for snapshot in snapshots:
        # Step 1: Position of each RNAP at this snapshot time.
        # Count how many positions have been exited (exit_time ≤ t AND > 0).
        # MATLAB: `sum(RNAP_exitTimes(:,:) <= t & RNAP_exitTimes(:,:) > 0, 1)`
        # Python axis=1: sum over positions (columns) for each RNAP (row).
        temp = np.sum((exits <= snapshot) & (exits > 0.0), axis=1)

        # Step 2: Filter mask — same three conditions as MATLAB.
        #   temp > 0         : RNAP has loaded and exited at least one position
        #   temp < gene_length: RNAP hasn't fully run off the gene yet
        #   max_per_rnap > snapshot: RNAP will still be transcribing at this time
        mask = (temp > 0) & (temp < gene_length) & (max_per_rnap > snapshot)

        # Step 3+4: Histogram active positions and accumulate.
        # MATLAB: histcounts(tempNETseq, 'BinMethod','integers', 'BinLimits',[1,geneLength])
        # np.bincount is equivalent for non-negative integer inputs.
        if np.any(mask):
            netseq_sum += np.bincount(temp[mask].astype(np.int64), minlength=gene_length + 1)[
                : gene_length + 1
            ]

    # Slice off index 0 (unused; biology is 1-indexed).
    # MATLAB output NETseq has rows 1..geneLength; Python returns a 0-indexed array
    # where element 0 corresponds to position 1 on the gene.
    return netseq_sum[1:]


# ══════════════════════════════════════════════════════════════════════════════
# Parameter helpers
# ══════════════════════════════════════════════════════════════════════════════
def _default_parameters() -> dict:
    """
    Default simulation parameters, identical to those in sjkimlab_NETSEQ_TASEP.m.

    MATLAB counterpart:
        parameters.RNAPSpeed  = 19;       % nt/s
        parameters.ribospeed  = 19;       % nt/s
        parameters.kLoading   = 1/20;     % 1/s; mean RNAP inter-arrival = 20 s
        parameters.kRiboLoading = 0;      % 1/s; 0 = no ribosomes
        parameters.KRutLoading  = 0.13;   % Rho/PT scaling factor
        parameters.simtime    = 2000;     % s
        parameters.glutime    = 1600;     % s; active transcription ends here
        parameters.geneLength = 3075;     % bp (overridden by dwell profile length)
        parameters.RNAP_dwellTimeProfile = ones(geneLength, 1);  % flat = uniform speed
    """
    return {
        "RNAPSpeed": 19,
        "ribospeed": 19,
        "kLoading": 1 / 20,
        "kRiboLoading": 0,
        "KRutLoading": 0.13,
        "simtime": 2000,
        "glutime": 1600,
        "geneLength": 3075,
        "RNAP_dwellTimeProfile": np.ones(3075),
    }


def _estimate_caps(
    gene_length: int,
    rnap_width: int,
    k_loading: float,
    simtime: int,
) -> tuple[int, int]:
    """
    Estimate biologically grounded active/history capacities for preallocation.

    MATLAB does not need this because arrays grow dynamically.  Numba requires
    fixed-size arrays, so we estimate upper bounds before compilation.

    active_cap : maximum number of RNAPs simultaneously on the gene.
                 Physical limit = floor(gene_length / RNAP_width), padded by
                 ACTIVE_CAP_SAFETY_MARGIN (1.25) for transient over-packing.

    history_cap: total number of RNAPs that could ever load over the full run.
                 Expected loads ≈ k_loading × simtime; padded 2× + 32 for safety.
                 Must also exceed active_cap + MIN_GAP so there's always room
                 for completed RNAPs in the history buffer.
    """
    packing_cap = max(1, int(math.ceil(gene_length / float(rnap_width))))
    active_cap = max(1, int(math.ceil(packing_cap * ACTIVE_CAP_SAFETY_MARGIN)))
    expected_loads = max(1.0, k_loading * simtime)
    history_cap = max(
        active_cap + HISTORY_CAP_MIN_GAP,
        int(math.ceil(expected_loads * HISTORY_CAP_BUFFER_FACTOR + HISTORY_CAP_BUFFER_ADD)),
    )
    return active_cap, history_cap


# ══════════════════════════════════════════════════════════════════════════════
# netseq_tasep_fast  — public API for a single simulation run
# ══════════════════════════════════════════════════════════════════════════════
def netseq_tasep_fast(input_parameters: dict | None = None, seed: int = 0) -> dict:
    """
    Run one fast TASEP simulation and return the snapshot-summed NET-seq signal.

    This is the Python equivalent of calling `sjkimlab_NETSEQ_TASEP(parameters)`
    once in MATLAB, but returns only the 7-snapshot sum rather than the full
    (geneLength × simtime) NETseq matrix.

    MATLAB workflow for a single run + NETseq extraction:
        output = sjkimlab_NETSEQ_TASEP(parameters);
        NETseqSum = sum(output.NETseq(:, 200:200:1400), 2);

    Python equivalent (one call to this function):
        result = netseq_tasep_fast(parameters, seed=42)
        NETseqSum = result["NETseq_sum"]  # shape: (gene_length,)

    Parameters
    ----------
    input_parameters : dict, optional
        Overrides for any default parameter (same keys as MATLAB struct fields).
        Only the keys you want to change need to be specified.
    seed : int
        RNG seed for reproducibility.  MATLAB uses a global random state;
        passing explicit seeds here enables reproducible parallel runs.

    Returns
    -------
    dict with keys:
        "parameters" : complete parameter dict (including derived values)
        "NETseq_sum" : np.ndarray of shape (gene_length,)
                       Sum of RNAP density at 7 snapshot times (200,400,...,1400 s),
                       averaged over those snapshots and expressed as a density profile.
    """
    # Merge user overrides into defaults (same logic as MATLAB's fieldnames loop).
    parameters = _default_parameters()
    if input_parameters:
        parameters.update(input_parameters)

    # Gene length is always derived from the dwell profile length, not geneLength.
    # MATLAB: parameters.geneLength = length(parameters.RNAP_dwellTimeProfile)
    dwell_profile = np.asarray(parameters["RNAP_dwellTimeProfile"], dtype=float).reshape(-1)
    gene_length = int(dwell_profile.size)
    parameters["geneLength"] = gene_length

    # rutSpeed = 5 × riboSpeed: Rho helicase moves 5× faster than the ribosome.
    # MATLAB: parameters.rutSpeed = 5 * parameters.ribospeed
    parameters["rutSpeed"] = 5 * float(parameters["ribospeed"])

    # Unpack scalar parameters for Numba (Numba cannot accept dicts).
    rnap_speed = float(parameters["RNAPSpeed"])
    ribo_speed = float(parameters["ribospeed"])
    simtime = int(parameters["simtime"])
    glutime = float(parameters["glutime"])
    k_loading = float(parameters["kLoading"])
    k_ribo_loading = float(parameters["kRiboLoading"])
    pt_percent = float(parameters["KRutLoading"])

    # Physical constants (same as MATLAB hardcoded values).
    rnap_width = 35        # RNAP footprint on DNA (bp); MATLAB: RNAP_width = 35
    dx = 1.0               # lattice spacing (bp); MATLAB: dx = 1
    dt = 0.1               # time step (s); MATLAB: dt = 0.1

    # minRholoadRNA: minimum exposed RNA before Rho can load.
    # MATLAB: minRholoadRNA = 80 - rho_width = 80 - 30 = 50
    min_rho_load_rna = 80 - 30

    # Look-ahead window sizes.
    # MATLAB: bases_evaluated = ceil(Speed * 10 * dt)
    # The factor of 10 provides a large enough window to capture all possible
    # crossings within one dt even at maximum speed.  At typical speed (19 nt/s)
    # and dt=0.1 s, bases_rnap = ceil(19 × 10 × 0.1) = ceil(19) = 19.
    bases_rnap = int(math.ceil(rnap_speed * 10.0 * dt))
    bases_ribo = int(math.ceil(ribo_speed * 10.0 * dt))

    # Rho window: MATLAB uses `riboSpeed*5` (= 5 × riboSpeed bases) as the
    # look-ahead, not `rutSpeed*10*dt`.  Both formulas evaluate to ~95 for
    # default parameters (riboSpeed=19, rutSpeed=95, dt=0.1).
    # MATLAB: `rho_locs(RNA)+riboSpeed*5 <= geneLength` as window upper bound.
    rho_window = max(1, int(round(ribo_speed * 5.0)))

    # ── Build per-position dwell time arrays ─────────────────────────────────
    # Index 0 is unused (gene is 1-indexed); positions 1..gene_length are valid.
    #
    # specific_dwelltime1[p] = mean dwell time of RNAP at position p (s/nt)
    # MATLAB: specificDwelltime1 = avgDwelltime1 .* RNAP_dwellTimeProfile
    #         where avgDwelltime1 = dx / RNAPSpeed
    specific_dwelltime1 = np.zeros(gene_length + 1, dtype=np.float64)
    specific_dwelltime1[1:] = (dx / rnap_speed) * dwell_profile

    # ribo_specific_dwelltime1[p] = mean ribosome dwell time per codon (s/nt), uniform.
    # MATLAB: RibospecificDwelltime1 = riboavgDwelltime * ones(geneLength, 1)
    #         where riboavgDwelltime = dx / riboSpeed
    ribo_specific_dwelltime1 = np.zeros(gene_length + 1, dtype=np.float64)
    ribo_specific_dwelltime1[1:] = dx / ribo_speed

    # specific_dwelltime_rho[p] = mean Rho dwell time per base (s/nt), uniform.
    # MATLAB: specificDwelltimeRho = dx/rutSpeed * ones(geneLength/dx, 1)
    specific_dwelltime_rho = np.zeros(gene_length + 1, dtype=np.float64)
    rut_speed = float(parameters["rutSpeed"])
    if rut_speed > 0.0:
        specific_dwelltime_rho[1:] = dx / rut_speed

    # ── Preallocation sizing ──────────────────────────────────────────────────
    # Estimates maximum number of active and total-ever-loaded RNAPs.
    # MATLAB skips this because its arrays grow dynamically.
    active_cap, history_cap = _estimate_caps(
        gene_length=gene_length,
        rnap_width=rnap_width,
        k_loading=k_loading,
        simtime=simtime,
    )
    parameters["activeCap"] = int(active_cap)
    parameters["historyCap"] = int(history_cap)

    # ── Run the Numba-compiled core ───────────────────────────────────────────
    # This call is equivalent to the entire `for t=0:dt:simtime` loop in MATLAB.
    # On the first call, Numba JIT-compiles _tasep_core (~2–5 s one-time cost).
    # Subsequent calls use the cached native code.
    rnap_exit_matrix, n_rnap = _tasep_core(
        seed=int(seed),
        active_cap=active_cap,
        history_capacity=history_cap,
        gene_length=gene_length,
        simtime=simtime,
        glutime=glutime,
        dt=dt,
        k_loading=k_loading,
        k_ribo_loading=k_ribo_loading,
        pt_percent=pt_percent,
        rnap_width=rnap_width,
        min_rho_load_rna=min_rho_load_rna,
        bases_rnap=bases_rnap,
        bases_ribo=bases_ribo,
        rho_window=rho_window,
        specific_dwelltime1=specific_dwelltime1,
        ribo_specific_dwelltime1=ribo_specific_dwelltime1,
        specific_dwelltime_rho=specific_dwelltime_rho,
    )

    # ── Compute snapshot-summed NET-seq signal ────────────────────────────────
    # Equivalent to MATLAB's post-simulation NETseq loop + NETSEQ_simulations.m:
    #   for t = 1:simtime: NETseq(:,t) = histcounts(...)
    #   NETseqSum = sum(NETseq(:, 200:200:1400), 2)
    netseq_sum = _compute_netseq_sum(
        rnap_exit_matrix=rnap_exit_matrix,
        n_rnap=n_rnap,
        gene_length=gene_length,
        snapshots=np.asarray(SNAPSHOTS, dtype=np.int64),
    )
    return {"parameters": parameters, "NETseq_sum": netseq_sum}


# ══════════════════════════════════════════════════════════════════════════════
# Gene data loader
# ══════════════════════════════════════════════════════════════════════════════
def _load_gene_parameters(gene: str) -> dict:
    """
    Load gene-specific parameters from CSV files.

    Equivalent to the data-loading section in NETSEQ_simulations.m:
        EcoligeneTE = readtable('Ecoli_gene_TE.csv');
        kribo = EcoligeneTE.TranslationEfficiencyAU(EcoligeneTE.Gene==gene) / 5;
        parameters.RNAP_dwellTimeProfile = readmatrix(genefname);
        parameters.RNAP_dwellTimeProfile ./= mean(parameters.RNAP_dwellTimeProfile);

    The kRiboLoading rate is derived as (Translation Efficiency AU) / 5.
    Division by 5 scales TE (arbitrary units) to a plausible ribosome arrival
    rate in units of 1/s.

    The dwell profile is normalized so its mean equals 1, ensuring that the
    baseline RNAP speed remains `RNAPSpeed` nt/s regardless of gene-specific
    shape variation.
    """
    script_dir = Path(__file__).resolve().parent
    te_path = script_dir / "Ecoli_gene_TE.csv"
    gene_profile_path = script_dir / "NETSEQ_gene" / f"NETSEQ_{gene}.csv"

    if not te_path.exists():
        raise FileNotFoundError(f"Missing {te_path}")
    if not gene_profile_path.exists():
        raise FileNotFoundError(f"Missing {gene_profile_path}")

    te_df = pd.read_csv(te_path)
    # Flexible column name matching (case-insensitive, strips non-alphanumeric chars).
    normalized = {"".join(ch for ch in col.lower() if ch.isalnum()): col for col in te_df.columns}
    gene_col = normalized.get("gene")
    te_col = normalized.get("translationefficiencyau")
    if not gene_col or not te_col:
        raise ValueError(
            "Ecoli_gene_TE.csv must contain columns for gene and translation efficiency."
        )

    gene_series = te_df[gene_col].astype(str).str.strip()
    te_values = te_df.loc[gene_series == gene, te_col]
    if te_values.empty or math.isnan(float(te_values.iloc[0])):
        # Gene not found or TE is NaN → no ribosome (same as MATLAB's isnan check).
        kribo = 0.0
    else:
        kribo = float(te_values.iloc[0]) / 5.0   # MATLAB: kribo = TE / 5

    # Load and normalize the RNAP dwell time profile.
    # MATLAB: parameters.RNAP_dwellTimeProfile ./= mean(parameters.RNAP_dwellTimeProfile)
    dwell_profile = np.loadtxt(gene_profile_path, delimiter=",")
    dwell_profile = dwell_profile / np.mean(dwell_profile)

    return {
        "KRutLoading": 0.13,
        "RNAP_dwellTimeProfile": dwell_profile,
        "kRiboLoading": kribo,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Worker function for parallel execution
# ══════════════════════════════════════════════════════════════════════════════
# MATLAB has no built-in equivalent: NETSEQ_simulations.m loops sequentially:
#   for i = 1:nloci: output = sjkimlab_NETSEQ_TASEP(parameters); ...
#
# Python uses ThreadPoolExecutor to run simulations in parallel across CPU cores.
# With nogil=True on the Numba kernels, threads achieve true parallelism without
# needing process-based isolation or parameter pickling.

def _worker(args: tuple[dict, int]) -> np.ndarray:
    """Single-run worker: run one simulation and return its NETseq_sum."""
    parameters, seed = args
    result = netseq_tasep_fast(parameters, seed=seed)
    return np.asarray(result["NETseq_sum"], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# run_netseq_simulations_fast  — public API for multi-run averaging
# ══════════════════════════════════════════════════════════════════════════════
def run_netseq_simulations_fast(
    gene: str,
    n_runs: int,
    seed: int | None = None,
    n_workers: int | None = None,
) -> np.ndarray:
    """
    Average exactly n_runs fast simulations and return the mean NETseq signal.

    MATLAB equivalent (from NETSEQ_simulations.m):
        final_output = sjkimlab_NETSEQ_TASEP(parameters);
        for i = 1:nloci
            output = sjkimlab_NETSEQ_TASEP(parameters);
            final_output.NETseq += output.NETseq;
        end
        final_output.NETseq /= nloci;
        NETseqSum = sum(final_output.NETseq(:, 200:200:1500), 2);

    Python differences:
    - Each run uses a distinct seed (base_seed + i) for reproducibility.
      MATLAB uses a shared global RNG state, so results are not reproducible.
    - With n_workers > 1, runs execute in parallel across CPU cores.
      MATLAB uses a single core (unless Parallel Computing Toolbox is used).
    - Returns the mean of the snapshot-summed NETseq (already summed over the
      7 snapshot times internally), whereas MATLAB sums snapshots after averaging.

    Parameters
    ----------
    gene      : gene name matching files in NETSEQ_gene/ and Ecoli_gene_TE.csv
    n_runs    : number of independent simulation replicates to average
    seed      : base RNG seed; run i uses seed (base_seed + i).
                If None, a random base seed is drawn.
    n_workers : number of parallel worker threads.
                Defaults to os.cpu_count().  Set to 1 to run serially.

    Returns
    -------
    np.ndarray of shape (gene_length,): mean NETseq sum over n_runs replicates.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    parameters = _load_gene_parameters(gene)

    # Assign one unique seed per run for full reproducibility.
    # MATLAB has no equivalent seed management.
    base_seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    seeds = [base_seed + i for i in range(n_runs)]

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        # Serial path: same logic as MATLAB's for-loop over nloci.
        outputs = [_worker((parameters, s)) for s in seeds]
        netseq_total = np.zeros_like(outputs[0], dtype=float)
        for output in outputs:
            netseq_total += output
    else:
        # Parallel path: threads call netseq_tasep_fast directly with shared
        # parameters.  Numba's nogil=True releases the GIL during the compiled
        # _tasep_core kernel, enabling true multi-core parallelism.
        worker_args = [(parameters, s) for s in seeds]
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            outputs = list(executor.map(_worker, worker_args))
        netseq_total = np.zeros_like(outputs[0], dtype=float)
        for output in outputs:
            netseq_total += output

    # Divide by n_runs to get the mean.
    # MATLAB: final_output.NETseq /= nloci
    return netseq_total / float(n_runs)
