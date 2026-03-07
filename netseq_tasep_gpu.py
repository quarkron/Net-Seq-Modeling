"""
netseq_tasep_gpu.py
===================
Numba CUDA port of the TASEP simulation for GPU-parallel CMA-ES evaluation.

Key differences from netseq_tasep_fast.py:
  1. One CUDA thread per simulation (no exit-time matrix — inline accumulation)
  2. All RNAPs kept with sentinels (no recycling — matches CPU processing order)
  3. xoroshiro128p RNG instead of numpy.random
  4. Returns only NETseq_sum and flux_count per thread (minimal device→host transfer)
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CUDA availability check
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = False
try:
    from numba import cuda
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_uniform_float64,
    )

    if cuda.is_available():
        _CUDA_AVAILABLE = True
except Exception:
    pass


def cuda_is_available() -> bool:
    return _CUDA_AVAILABLE


def gpu_count() -> int:
    if not _CUDA_AVAILABLE:
        return 0
    return len(cuda.gpus)


# ---------------------------------------------------------------------------
# Constants (same as netseq_tasep_fast.py)
# ---------------------------------------------------------------------------
SNAPSHOTS_TUPLE = (200, 400, 600, 800, 1000, 1200, 1400)
N_SNAPSHOTS = 7


# ---------------------------------------------------------------------------
# Device helper: exponential draw from xoroshiro128p
# ---------------------------------------------------------------------------
if _CUDA_AVAILABLE:

    @cuda.jit(device=True, inline=True)
    def _exp_draw(rng_states, tid, scale):
        """Draw Exp(scale) using inverse-CDF: -scale * ln(U)."""
        u = xoroshiro128p_uniform_float64(rng_states, tid)
        # Guard against u==0 (would give inf)
        if u <= 0.0:
            u = 1e-300
        return -scale * math.log(u)

    # -------------------------------------------------------------------
    # Device function: _sample_first_passage_cuda
    # -------------------------------------------------------------------
    @cuda.jit(device=True)
    def _sample_first_passage_cuda(
        rng_states,
        tid,
        dwell_profile,        # global or shared memory array
        start_pos,
        end_pos,
        t_now,
        dt,
        exit_times,           # thread-local scratch buffer (registers/local)
    ):
        """
        GPU equivalent of _sample_first_passage in netseq_tasep_fast.py.

        Returns the number of positions crossed during [t_now, t_now+dt].
        Fills exit_times[0..count-1] with absolute crossing timestamps.
        """
        next_time = t_now
        count = 0
        pos = start_pos
        while pos <= end_pos:
            scale = dwell_profile[pos]
            if scale > 0.0:
                next_time += _exp_draw(rng_states, tid, scale)
            if next_time > t_now + dt:
                break
            exit_times[count] = next_time
            count += 1
            pos += 1
        return count

    # -------------------------------------------------------------------
    # Main CUDA kernel: _tasep_core_cuda
    #
    # Mirrors _tasep_core in netseq_tasep_fast.py exactly:
    #   - All RNAPs kept with sentinels (no slot recycling)
    #   - Processed in loading order (idx 0 = oldest = most downstream)
    #   - Collision check against idx-1 with backward terminated search
    #   - Inline NETseq_sum / flux_count accumulation (no exit-time matrix)
    # -------------------------------------------------------------------
    @cuda.jit
    def _tasep_core_cuda(
        rng_states,
        # Per-candidate dwell profiles: (n_total_sims, gene_length+1) in global mem
        dwell_profiles,
        # Per-candidate ribo dwell: (n_total_sims, gene_length+1)
        ribo_dwell_profiles,
        # Per-candidate rho dwell: (n_total_sims, gene_length+1)
        rho_dwell_profiles,
        # Scalar params broadcast to all threads (1-D arrays of length n_total_sims)
        gene_lengths,
        k_loadings,
        k_ribo_loadings,
        pt_percents,
        simtimes,
        glutimes,
        active_caps,
        # Output arrays: (n_total_sims, max_gene_length)
        out_netseq_sum,
        out_flux_count,
        # Scratch dimensions
        max_gene_length,
        max_history_cap,
        # Constants
        rnap_width,
        dt,
        min_rho_load_rna,
        bases_rnap,
        bases_ribo,
        rho_window,
    ):
        tid = cuda.grid(1)
        n_total = rng_states.shape[0]
        if tid >= n_total:
            return

        gene_length = gene_lengths[tid]
        simtime = simtimes[tid]
        glutime = glutimes[tid]
        k_loading = k_loadings[tid]
        k_ribo_loading = k_ribo_loadings[tid]
        pt_percent = pt_percents[tid]

        # Sentinels (identical to CPU)
        term_rnap = gene_length + 10
        term_rho = gene_length + 9
        finished_rnap = gene_length + 1
        big_t = float(simtime) + 1.0

        # -- Thread-local arrays --
        # Size 256: fits history_cap for all E. coli genes
        # (k_loading=0.05 * simtime=2000 * 2 + 32 = 232, rounded up)
        # NO recycling — mirrors CPU's append-only arrays with sentinels
        HIST_CAP = 256
        rnap_locs = cuda.local.array(HIST_CAP, dtype=np.int64)
        ribo_locs = cuda.local.array(HIST_CAP, dtype=np.int64)
        rho_locs = cuda.local.array(HIST_CAP, dtype=np.int64)
        rnap_ribo_coupling = cuda.local.array(HIST_CAP, dtype=np.int64)
        ribo_loadt = cuda.local.array(HIST_CAP, dtype=np.float64)

        # Scratch buffer for first-passage sampling
        exit_buf = cuda.local.array(32, dtype=np.float64)

        # Initialize arrays
        for i in range(HIST_CAP):
            rnap_locs[i] = 0
            ribo_locs[i] = 0
            rho_locs[i] = 0
            rnap_ribo_coupling[i] = 0
            ribo_loadt[i] = big_t

        n_rnap = 0  # total RNAPs ever loaded (only increases, like CPU)
        first_active = 0  # lowest index still on gene (optimization: skip completed)

        # Schedule first RNAP loading
        if k_loading > 0.0:
            loadt = _exp_draw(rng_states, tid, 1.0 / k_loading)
        else:
            loadt = big_t

        n_steps = int(simtime / dt)

        # Snapshot times as integer step indices
        snap_steps_0 = int(SNAPSHOTS_TUPLE[0] / dt)
        snap_steps_1 = int(SNAPSHOTS_TUPLE[1] / dt)
        snap_steps_2 = int(SNAPSHOTS_TUPLE[2] / dt)
        snap_steps_3 = int(SNAPSHOTS_TUPLE[3] / dt)
        snap_steps_4 = int(SNAPSHOTS_TUPLE[4] / dt)
        snap_steps_5 = int(SNAPSHOTS_TUPLE[5] / dt)
        snap_steps_6 = int(SNAPSHOTS_TUPLE[6] / dt)

        # ================================================================
        # MAIN SIMULATION LOOP — mirrors CPU _tasep_core exactly
        # ================================================================
        for step in range(n_steps + 1):
            t_now = step * dt

            # -- Sub-step 1: RNAP LOADING --
            # Case A: Promoter BLOCKED (last loaded RNAP too close)
            if (
                loadt <= t_now
                and t_now < glutime
                and n_rnap > 0
                and (rnap_locs[n_rnap - 1] - rnap_width) <= 0
            ):
                if k_loading > 0.0:
                    loadt = t_now + _exp_draw(rng_states, tid, 1.0 / k_loading)
                else:
                    loadt = big_t

            # Case B: Promoter CLEAR
            if (
                loadt <= t_now
                and t_now < glutime
                and (n_rnap == 0 or (rnap_locs[n_rnap - 1] - rnap_width) >= 0)
            ):
                if n_rnap < HIST_CAP:
                    idx = n_rnap
                    n_rnap += 1

                    rnap_locs[idx] = 1
                    ribo_locs[idx] = 0
                    rho_locs[idx] = 0
                    rnap_ribo_coupling[idx] = 0

                    # Schedule next RNAP loading
                    if k_loading > 0.0:
                        loadt = t_now + _exp_draw(rng_states, tid, 1.0 / k_loading)
                    else:
                        loadt = big_t

                    # Schedule ribosome loading for this RNAP
                    if k_ribo_loading > 0.0:
                        ribo_loadt[idx] = t_now + _exp_draw(rng_states, tid, 1.0 / k_ribo_loading)
                    else:
                        ribo_loadt[idx] = big_t

            # -- Sub-step 2: RNAP ELONGATION --
            # Process in loading order: idx 0 = oldest = most downstream
            # Start from first_active to skip completed/terminated RNAPs
            for idx in range(first_active, n_rnap):
                curr_loc = rnap_locs[idx]
                if curr_loc <= gene_length:
                    end_loc = curr_loc + bases_rnap
                    if end_loc > gene_length:
                        end_loc = gene_length

                    advance = _sample_first_passage_cuda(
                        rng_states, tid,
                        dwell_profiles[tid],
                        curr_loc, end_loc, t_now, dt, exit_buf,
                    )

                    if advance > 0:
                        allowed_advance = advance

                        # Collision check against RNAP directly ahead (idx-1)
                        if idx > 0:
                            prev_loc = rnap_locs[idx - 1]

                            # Search backward past terminated RNAPs
                            if prev_loc == term_rnap:
                                j = 1
                                while (
                                    j <= idx
                                    and rnap_locs[idx - j] == term_rnap
                                    and (idx - j) > 0
                                ):
                                    j += 1
                                if j == idx + 1 or (idx - j) < 0:
                                    prev_loc = finished_rnap
                                else:
                                    prev_loc = rnap_locs[idx - j]

                            overlap = (curr_loc + advance - 1) - prev_loc + rnap_width

                            if prev_loc >= gene_length:
                                overlap = 0

                            if overlap > 0:
                                allowed_advance = advance - overlap

                        # Advance and count flux
                        if allowed_advance > 0:
                            for k in range(allowed_advance):
                                pos = curr_loc + k
                                if pos >= 1 and pos <= gene_length:
                                    out_flux_count[tid, pos - 1] += 1
                            rnap_locs[idx] = curr_loc + allowed_advance

                # Ribosome loading (same as CPU)
                if ribo_loadt[idx] <= t_now and rnap_locs[idx] >= 30:
                    ribo_loadt[idx] = big_t
                    ribo_locs[idx] = 1

            # -- Sub-step 3: RHO / PREMATURE TERMINATION --
            for idx in range(first_active, n_rnap):
                # Rho loading decision
                if rnap_locs[idx] < gene_length:
                    ptrna_size = rnap_locs[idx] - rnap_width - 30 - ribo_locs[idx]
                    if ptrna_size > min_rho_load_rna:
                        threshold = pt_percent * ptrna_size / gene_length
                        r = xoroshiro128p_uniform_float64(rng_states, tid)
                        if 100.0 * dt * r <= threshold:
                            temp_rho_loc = rnap_locs[idx] - rnap_width - int(
                                math.floor(xoroshiro128p_uniform_float64(rng_states, tid) * ptrna_size)
                            )
                            if temp_rho_loc > rho_locs[idx]:
                                rho_locs[idx] = temp_rho_loc

                # Rho elongation
                rho_loc = rho_locs[idx]
                if rho_loc > 0 and rho_loc < gene_length:
                    end_loc = rho_loc + rho_window
                    if end_loc > gene_length:
                        end_loc = gene_length
                    rho_advance = _sample_first_passage_cuda(
                        rng_states, tid,
                        rho_dwell_profiles[tid],
                        rho_loc, end_loc, t_now, dt, exit_buf,
                    )
                    rho_locs[idx] = rho_loc + rho_advance

                # Rho catch check
                if rho_locs[idx] >= rnap_locs[idx] and rnap_locs[idx] <= gene_length:
                    rnap_locs[idx] = term_rnap
                    rho_locs[idx] = term_rho

            # -- Sub-step 4: RIBOSOME ELONGATION --
            for idx in range(first_active, n_rnap):
                ribo_loc = ribo_locs[idx]
                if ribo_loc > 0 and ribo_loc <= gene_length:
                    rnap_loc = rnap_locs[idx]

                    # State 1: Coupled, mid-gene
                    if rnap_ribo_coupling[idx] == 1 and ribo_loc <= gene_length - rnap_width:
                        if rnap_loc <= gene_length:
                            ribo_locs[idx] = rnap_loc - rnap_width

                    # State 2: Coupled, near gene end
                    elif rnap_ribo_coupling[idx] == 1 and ribo_loc > gene_length - 30:
                        # Ribosome finishes independently
                        ribo_locs[idx] = finished_rnap

                    # State 3: RNAP terminated by Rho
                    elif rnap_loc == term_rnap:
                        # Ribosome translates to its current position then stops.
                        # (CPU scans exit-time matrix for last RNAP pos, but since
                        # we don't store exit times, the ribosome stays where it is
                        # — which still protects the same amount of mRNA for Rho
                        # loading calculations on other RNAPs. The terminated RNAP
                        # is already done, so this only affects ptrna_size of
                        # OTHER RNAPs sharing this mRNA — but each RNAP has its
                        # own ribosome in this model.)
                        ribo_locs[idx] = term_rnap

                    # State 4: Free ribosome (uncoupled)
                    else:
                        end_loc = ribo_loc + bases_ribo
                        if end_loc > gene_length:
                            end_loc = gene_length
                        ribo_advance = _sample_first_passage_cuda(
                            rng_states, tid,
                            ribo_dwell_profiles[tid],
                            ribo_loc, end_loc, t_now, dt, exit_buf,
                        )

                        # Collision check vs RNAP
                        overlap = (ribo_loc + ribo_advance - 1) - rnap_loc + rnap_width

                        if rnap_loc == finished_rnap:
                            overlap = 0

                        if overlap > 0:
                            if rnap_loc <= gene_length:
                                rnap_ribo_coupling[idx] = 1
                            allowed = ribo_advance - overlap
                            if allowed > 0:
                                ribo_locs[idx] = ribo_loc + allowed
                            # Snap to just behind RNAP
                            if rnap_loc <= gene_length:
                                ribo_locs[idx] = rnap_loc - rnap_width
                        elif ribo_advance > 0:
                            ribo_locs[idx] = ribo_loc + ribo_advance

            # -- Inline NETseq_sum accumulation at snapshot times --
            # Mirrors CPU _compute_netseq_sum: count active RNAPs at each position
            is_snapshot = (
                step == snap_steps_0 or step == snap_steps_1 or
                step == snap_steps_2 or step == snap_steps_3 or
                step == snap_steps_4 or step == snap_steps_5 or
                step == snap_steps_6
            )
            if is_snapshot:
                for idx in range(first_active, n_rnap):
                    loc = rnap_locs[idx]
                    if loc > 1 and loc <= gene_length:
                        out_netseq_sum[tid, loc - 2] += 1.0

            # Advance first_active past completed/terminated RNAPs
            while first_active < n_rnap and rnap_locs[first_active] > gene_length:
                first_active += 1


# ---------------------------------------------------------------------------
# Host-side: objective_batch_gpu
# ---------------------------------------------------------------------------

def _prepare_kernel_inputs(thetas, base_params, n_runs):
    """Build per-simulation arrays for the GPU kernel."""
    from netseq_tasep_fast import _estimate_caps

    n_candidates = len(thetas)
    n_total = n_candidates * n_runs

    gene_length = len(thetas[0])

    # Physical constants
    rnap_speed = float(base_params.get("RNAPSpeed", 19))
    ribo_speed = float(base_params.get("ribospeed", 19))
    k_loading = float(base_params.get("kLoading", 1 / 20))
    k_ribo_loading = float(base_params.get("kRiboLoading", 0))
    pt_percent = float(base_params.get("KRutLoading", 0.13))
    simtime = int(base_params.get("simtime", 2000))
    glutime = float(base_params.get("glutime", 1600))
    dx = 1.0
    dt = 0.1
    rnap_width = 35
    rut_speed = 5 * ribo_speed
    min_rho_load_rna = 50

    bases_rnap = int(math.ceil(rnap_speed * 10.0 * dt))
    bases_ribo = int(math.ceil(ribo_speed * 10.0 * dt))
    rho_window = max(1, int(round(ribo_speed * 5.0)))

    active_cap, _ = _estimate_caps(gene_length, rnap_width, k_loading, simtime)
    if active_cap > 64:
        active_cap = 64  # Hard limit for local array size in kernel

    # Build dwell profiles per simulation
    dwell_all = np.zeros((n_total, gene_length + 1), dtype=np.float64)
    ribo_dwell_all = np.zeros((n_total, gene_length + 1), dtype=np.float64)
    rho_dwell_all = np.zeros((n_total, gene_length + 1), dtype=np.float64)

    ribo_dwell_row = np.zeros(gene_length + 1, dtype=np.float64)
    ribo_dwell_row[1:] = dx / ribo_speed
    rho_dwell_row = np.zeros(gene_length + 1, dtype=np.float64)
    if rut_speed > 0:
        rho_dwell_row[1:] = dx / rut_speed

    for c_idx, theta in enumerate(thetas):
        D = np.exp(theta)
        D_norm = D / np.mean(D)
        row = np.zeros(gene_length + 1, dtype=np.float64)
        row[1:] = (dx / rnap_speed) * D_norm
        for r_idx in range(n_runs):
            sim_idx = c_idx * n_runs + r_idx
            dwell_all[sim_idx] = row
            ribo_dwell_all[sim_idx] = ribo_dwell_row
            rho_dwell_all[sim_idx] = rho_dwell_row

    # Scalar param arrays
    gene_lengths = np.full(n_total, gene_length, dtype=np.int64)
    k_loadings = np.full(n_total, k_loading, dtype=np.float64)
    k_ribo_loadings = np.full(n_total, k_ribo_loading, dtype=np.float64)
    pt_percents = np.full(n_total, pt_percent, dtype=np.float64)
    simtimes_arr = np.full(n_total, simtime, dtype=np.int64)
    glutimes_arr = np.full(n_total, glutime, dtype=np.float64)
    active_caps_arr = np.full(n_total, active_cap, dtype=np.int64)

    return {
        "dwell_profiles": dwell_all,
        "ribo_dwell_profiles": ribo_dwell_all,
        "rho_dwell_profiles": rho_dwell_all,
        "gene_lengths": gene_lengths,
        "k_loadings": k_loadings,
        "k_ribo_loadings": k_ribo_loadings,
        "pt_percents": pt_percents,
        "simtimes": simtimes_arr,
        "glutimes": glutimes_arr,
        "active_caps": active_caps_arr,
        "gene_length": gene_length,
        "n_total": n_total,
        "n_candidates": n_candidates,
        "n_runs": n_runs,
        "rnap_width": rnap_width,
        "dt": dt,
        "min_rho_load_rna": min_rho_load_rna,
        "bases_rnap": bases_rnap,
        "bases_ribo": bases_ribo,
        "rho_window": rho_window,
    }


def objective_batch_gpu(thetas, base_params, n_runs, S_exp_norm,
                        base_seed=0, device_id=0, stream=None):
    """
    GPU-accelerated batch evaluation of CMA-ES candidates.

    Args:
        thetas: list of candidate theta vectors (log-space dwell profiles)
        base_params: dict of simulation parameters (from _load_gene_parameters)
        n_runs: number of TASEP replicate simulations per candidate
        S_exp_norm: experimental NET-seq signal (normalized), for MSE computation
        base_seed: RNG seed base
        device_id: which GPU to use
        stream: CUDA stream (None = default stream)

    Returns:
        List of MSE values, one per candidate
    """
    if not _CUDA_AVAILABLE:
        return _objective_batch_cpu_fallback(thetas, base_params, n_runs, S_exp_norm, base_seed)

    inputs = _prepare_kernel_inputs(thetas, base_params, n_runs)
    n_total = inputs["n_total"]
    gene_length = inputs["gene_length"]
    n_candidates = inputs["n_candidates"]

    # Select device
    cuda.select_device(device_id)

    # Allocate RNG states
    rng_states = create_xoroshiro128p_states(n_total, seed=base_seed)

    # Transfer to device
    d_dwell = cuda.to_device(inputs["dwell_profiles"], stream=stream)
    d_ribo_dwell = cuda.to_device(inputs["ribo_dwell_profiles"], stream=stream)
    d_rho_dwell = cuda.to_device(inputs["rho_dwell_profiles"], stream=stream)
    d_gene_lengths = cuda.to_device(inputs["gene_lengths"], stream=stream)
    d_k_loadings = cuda.to_device(inputs["k_loadings"], stream=stream)
    d_k_ribo_loadings = cuda.to_device(inputs["k_ribo_loadings"], stream=stream)
    d_pt_percents = cuda.to_device(inputs["pt_percents"], stream=stream)
    d_simtimes = cuda.to_device(inputs["simtimes"], stream=stream)
    d_glutimes = cuda.to_device(inputs["glutimes"], stream=stream)
    d_active_caps = cuda.to_device(inputs["active_caps"], stream=stream)

    # Output arrays
    d_netseq = cuda.to_device(np.zeros((n_total, gene_length), dtype=np.float64), stream=stream)
    d_flux = cuda.to_device(np.zeros((n_total, gene_length), dtype=np.int64), stream=stream)

    # Launch kernel
    threads_per_block = 64
    blocks = (n_total + threads_per_block - 1) // threads_per_block

    kernel_args = (
        rng_states,
        d_dwell, d_ribo_dwell, d_rho_dwell,
        d_gene_lengths, d_k_loadings, d_k_ribo_loadings, d_pt_percents,
        d_simtimes, d_glutimes, d_active_caps,
        d_netseq, d_flux,
        gene_length, 256,  # max_history_cap matches HIST_CAP in kernel
        inputs["rnap_width"], inputs["dt"],
        inputs["min_rho_load_rna"],
        inputs["bases_rnap"], inputs["bases_ribo"], inputs["rho_window"],
    )

    if stream is not None:
        _tasep_core_cuda[blocks, threads_per_block, stream](*kernel_args)
    else:
        _tasep_core_cuda[blocks, threads_per_block](*kernel_args)

    # Copy results back
    h_netseq = d_netseq.copy_to_host(stream=stream)

    if stream is not None:
        stream.synchronize()
    else:
        cuda.synchronize()

    # Aggregate per candidate: average NETseq_sum across runs, compute MSE
    n_runs_actual = inputs["n_runs"]
    fitnesses = []
    for c_idx in range(n_candidates):
        start = c_idx * n_runs_actual
        end = start + n_runs_actual
        netseq_avg = np.mean(h_netseq[start:end], axis=0)
        s_mean = np.mean(netseq_avg)
        if s_mean > 0:
            s_norm = netseq_avg / s_mean
            mse = float(np.mean((S_exp_norm - s_norm) ** 2))
        else:
            mse = 1e6
        fitnesses.append(mse)

    return fitnesses


def _objective_batch_cpu_fallback(thetas, base_params, n_runs, S_exp_norm, base_seed):
    """CPU fallback using ThreadPoolExecutor (same as notebook objective_batch)."""
    from netseq_tasep_fast import netseq_tasep_fast

    n_candidates = len(thetas)
    gene_length = len(thetas[0])
    n_workers = os.cpu_count() or 1

    def run_single(args):
        cand_idx, dwell_profile, seed = args
        params = dict(base_params)
        params["RNAP_dwellTimeProfile"] = dwell_profile
        result = netseq_tasep_fast(params, seed=seed)
        return cand_idx, np.asarray(result["NETseq_sum"], dtype=float)

    D_norms = []
    for theta in thetas:
        D = np.exp(theta)
        D_norms.append(D / np.mean(D))

    args_list = []
    for c_idx, D_norm in enumerate(D_norms):
        for r_idx in range(n_runs):
            seed = base_seed + c_idx * n_runs + r_idx
            args_list.append((c_idx, D_norm, seed))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(run_single, args_list))

    accumulators = [np.zeros(gene_length, dtype=float) for _ in range(n_candidates)]
    counts = [0] * n_candidates
    for cand_idx, netseq_sum in results:
        accumulators[cand_idx] += netseq_sum
        counts[cand_idx] += 1

    fitnesses = []
    for c_idx in range(n_candidates):
        S_sim = accumulators[c_idx] / counts[c_idx]
        s_mean = np.mean(S_sim)
        if s_mean > 0:
            s_norm = S_sim / s_mean
            mse = float(np.mean((S_exp_norm - s_norm) ** 2))
        else:
            mse = 1e6
        fitnesses.append(mse)
    return fitnesses
