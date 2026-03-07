"""
netseq_tasep_gpu.py
===================
Numba CUDA port of the TASEP simulation for GPU-parallel CMA-ES evaluation.

v3.0 — GPU optimizations (Phases 1+2):
  - Float32 arithmetic (32x FP32:FP64 throughput ratio on Ampere)
  - Int32 thread-local arrays (halved local memory per thread)
  - Dwell profile deduplication (n_candidates rows, not n_total)
  - Optional flux computation (disabled during CMA-ES optimization)
  - GPU-side atomic reduction (n_candidates output rows, not n_total)
  - GPU-side MSE kernel with block-level reduction (minimal D2H transfer)
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
        xoroshiro128p_uniform_float32,
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
# CUDA kernels (only compiled when CUDA is available)
# ---------------------------------------------------------------------------
if _CUDA_AVAILABLE:

    # -----------------------------------------------------------------------
    # Device helper: exponential draw (float32 RNG)
    # -----------------------------------------------------------------------
    @cuda.jit(device=True, inline=True)
    def _exp_draw(rng_states, tid, scale):
        """Draw Exp(scale) using inverse-CDF: -scale * ln(U)."""
        u = xoroshiro128p_uniform_float32(rng_states, tid)
        if u < 1e-30:
            u = 1e-30
        return -scale * math.log(u)

    # -----------------------------------------------------------------------
    # Device function: _sample_first_passage_cuda
    # -----------------------------------------------------------------------
    @cuda.jit(device=True)
    def _sample_first_passage_cuda(
        rng_states,
        tid,
        dwell_profile,        # global memory array
        start_pos,
        end_pos,
        t_now,
        dt,
        exit_times,           # thread-local scratch buffer
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

    # -----------------------------------------------------------------------
    # Main CUDA kernel: _tasep_core_cuda (v3.0)
    #
    # Changes from v2.2:
    #   - Float32 local arrays + RNG (halved local memory per thread)
    #   - Deduplicated dwell profiles: reads dwell_profiles[candidate_idx]
    #   - Scalar physical params (not per-thread arrays)
    #   - Atomic reduction: writes to out_netseq_sum[candidate_idx, ...]
    #   - Optional flux via compute_flux flag
    # -----------------------------------------------------------------------
    @cuda.jit
    def _tasep_core_cuda(
        rng_states,
        # Deduplicated dwell profiles: (n_candidates, gene_length+1) float32
        dwell_profiles,
        # Shared ribo dwell: (gene_length+1,) float32
        ribo_dwell_profiles,
        # Shared rho dwell: (gene_length+1,) float32
        rho_dwell_profiles,
        # Output: (n_candidates, gene_length) float32 — atomic reduction
        out_netseq_sum,
        # Output: (n_candidates, gene_length) int32, or (1,1) dummy
        out_flux_count,
        # Simulation dimensions
        n_runs,           # int32: threads per candidate
        gene_length,      # int32
        compute_flux,     # int32: 0=skip flux, 1=compute flux
        # Scalar physical params (shared by all threads)
        k_loading,        # float32
        k_ribo_loading,   # float32
        pt_percent,       # float32
        simtime,          # int32
        glutime,          # float32
        # Constants
        rnap_width,       # int32
        dt,               # float32
        min_rho_load_rna, # int32
        bases_rnap,       # int32
        bases_ribo,       # int32
        rho_window,       # int32
    ):
        tid = cuda.grid(1)
        n_total = rng_states.shape[0]
        if tid >= n_total:
            return

        candidate_idx = tid // n_runs

        # Sentinels (identical to CPU)
        term_rnap = gene_length + 10
        term_rho = gene_length + 9
        finished_rnap = gene_length + 1
        big_t = float(simtime) + 1.0

        # -- Thread-local arrays (int32/float32 for reduced local memory) --
        # Size 128: fits history_cap for all E. coli genes
        # (k_loading=0.05 * simtime=2000 * 1.2 = 120, rounded up)
        HIST_CAP = 128
        rnap_locs = cuda.local.array(HIST_CAP, dtype=np.int32)
        ribo_locs = cuda.local.array(HIST_CAP, dtype=np.int32)
        rho_locs = cuda.local.array(HIST_CAP, dtype=np.int32)
        rnap_ribo_coupling = cuda.local.array(HIST_CAP, dtype=np.int32)
        ribo_loadt = cuda.local.array(HIST_CAP, dtype=np.float32)

        # Scratch buffer for first-passage sampling
        exit_buf = cuda.local.array(32, dtype=np.float32)

        # Initialize arrays
        for i in range(HIST_CAP):
            rnap_locs[i] = 0
            ribo_locs[i] = 0
            rho_locs[i] = 0
            rnap_ribo_coupling[i] = 0
            ribo_loadt[i] = big_t

        n_rnap = 0  # total RNAPs ever loaded (only increases, like CPU)
        first_active = 0  # lowest index still on gene

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
            for idx in range(first_active, n_rnap):
                curr_loc = rnap_locs[idx]
                if curr_loc <= gene_length:
                    end_loc = curr_loc + bases_rnap
                    if end_loc > gene_length:
                        end_loc = gene_length

                    advance = _sample_first_passage_cuda(
                        rng_states, tid,
                        dwell_profiles[candidate_idx],
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

                        # Advance and optionally count flux
                        if allowed_advance > 0:
                            if compute_flux:
                                for k in range(allowed_advance):
                                    pos = curr_loc + k
                                    if pos >= 1 and pos <= gene_length:
                                        cuda.atomic.add(out_flux_count, (candidate_idx, pos - 1), 1)
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
                        r = xoroshiro128p_uniform_float32(rng_states, tid)
                        if 100.0 * dt * r <= threshold:
                            temp_rho_loc = rnap_locs[idx] - rnap_width - int(
                                math.floor(xoroshiro128p_uniform_float32(rng_states, tid) * ptrna_size)
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
                        rho_dwell_profiles,
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
                        ribo_locs[idx] = finished_rnap

                    # State 3: RNAP terminated by Rho
                    elif rnap_loc == term_rnap:
                        ribo_locs[idx] = term_rnap

                    # State 4: Free ribosome (uncoupled)
                    else:
                        end_loc = ribo_loc + bases_ribo
                        if end_loc > gene_length:
                            end_loc = gene_length
                        ribo_advance = _sample_first_passage_cuda(
                            rng_states, tid,
                            ribo_dwell_profiles,
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

            # -- Snapshot: atomic reduction to (n_candidates, gene_length) --
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
                        cuda.atomic.add(out_netseq_sum, (candidate_idx, loc - 2), np.float32(1.0))

            # Advance first_active past completed/terminated RNAPs
            while first_active < n_rnap and rnap_locs[first_active] > gene_length:
                first_active += 1

    # -------------------------------------------------------------------
    # MSE reduction kernel (one thread block per candidate)
    #
    # Uses shared-memory block reduction (inspired by warp-shuffle
    # reduction patterns from GPU fluid simulation literature).
    # -------------------------------------------------------------------
    @cuda.jit
    def _compute_mse_cuda(
        netseq_sum,    # (n_candidates, gene_length) float32 — raw counts
        S_exp_norm,    # (gene_length,) float32 — experimental signal
        mse_out,       # (n_candidates,) float32 — output MSE per candidate
        n_runs,        # int32
        gene_length,   # int32
        n_candidates,  # int32
    ):
        c_idx = cuda.blockIdx.x
        if c_idx >= n_candidates:
            return
        t = cuda.threadIdx.x
        block_size = cuda.blockDim.x

        sdata = cuda.shared.array(256, dtype=np.float32)

        # Step 1: sum netseq_sum[c_idx, :] for normalization
        local_sum = np.float32(0.0)
        pos = t
        while pos < gene_length:
            local_sum += netseq_sum[c_idx, pos]
            pos += block_size
        sdata[t] = local_sum
        cuda.syncthreads()

        s = block_size // 2
        while s > 0:
            if t < s:
                sdata[t] += sdata[t + s]
            cuda.syncthreads()
            s //= 2

        total_sum = sdata[0]
        cuda.syncthreads()

        # s_mean = total_sum / (n_runs * gene_length)
        denom = np.float32(n_runs) * np.float32(gene_length)
        s_mean = total_sum / denom

        if s_mean <= np.float32(0.0):
            if t == 0:
                mse_out[c_idx] = np.float32(1e6)
            return

        # Step 2: MSE = mean((S_exp_norm[pos] - sim_norm[pos])^2)
        norm_factor = np.float32(n_runs) * s_mean
        local_mse = np.float32(0.0)
        pos = t
        while pos < gene_length:
            sim_norm = netseq_sum[c_idx, pos] / norm_factor
            diff = S_exp_norm[pos] - sim_norm
            local_mse += diff * diff
            pos += block_size
        sdata[t] = local_mse
        cuda.syncthreads()

        s = block_size // 2
        while s > 0:
            if t < s:
                sdata[t] += sdata[t + s]
            cuda.syncthreads()
            s //= 2

        if t == 0:
            mse_out[c_idx] = sdata[0] / np.float32(gene_length)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------

def _prepare_kernel_inputs(thetas, base_params, n_runs):
    """Build deduplicated arrays and scalar params for the GPU kernel."""
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

    # Deduplicated dwell profiles: (n_candidates, gene_length+1) float32
    dwell_all = np.zeros((n_candidates, gene_length + 1), dtype=np.float32)

    for c_idx, theta in enumerate(thetas):
        D = np.exp(theta)
        D_norm = D / np.mean(D)
        dwell_all[c_idx, 1:] = ((dx / rnap_speed) * D_norm).astype(np.float32)

    # ribo/rho dwell are identical for all sims — single 1D row
    ribo_dwell_all = np.zeros(gene_length + 1, dtype=np.float32)
    ribo_dwell_all[1:] = np.float32(dx / ribo_speed)
    rho_dwell_all = np.zeros(gene_length + 1, dtype=np.float32)
    if rut_speed > 0:
        rho_dwell_all[1:] = np.float32(dx / rut_speed)

    return {
        "dwell_profiles": dwell_all,
        "ribo_dwell_profiles": ribo_dwell_all,
        "rho_dwell_profiles": rho_dwell_all,
        "gene_length": gene_length,
        "n_total": n_total,
        "n_candidates": n_candidates,
        "n_runs": n_runs,
        # Scalar params (typed for kernel)
        "k_loading": np.float32(k_loading),
        "k_ribo_loading": np.float32(k_ribo_loading),
        "pt_percent": np.float32(pt_percent),
        "simtime": np.int32(simtime),
        "glutime": np.float32(glutime),
        "rnap_width": np.int32(rnap_width),
        "dt": np.float32(dt),
        "min_rho_load_rna": np.int32(min_rho_load_rna),
        "bases_rnap": np.int32(bases_rnap),
        "bases_ribo": np.int32(bases_ribo),
        "rho_window": np.int32(rho_window),
    }


def _launch_tasep_kernel(inputs, rng_states, d_dwell, d_ribo_dwell, d_rho_dwell,
                         d_netseq, d_flux, compute_flux, stream):
    """Launch the TASEP kernel with the standard argument layout."""
    n_total = inputs["n_total"]
    threads_per_block = 128
    blocks = (n_total + threads_per_block - 1) // threads_per_block

    kernel_args = (
        rng_states,
        d_dwell, d_ribo_dwell, d_rho_dwell,
        d_netseq, d_flux,
        np.int32(inputs["n_runs"]), np.int32(inputs["gene_length"]),
        np.int32(compute_flux),
        inputs["k_loading"], inputs["k_ribo_loading"], inputs["pt_percent"],
        inputs["simtime"], inputs["glutime"],
        inputs["rnap_width"], inputs["dt"], inputs["min_rho_load_rna"],
        inputs["bases_rnap"], inputs["bases_ribo"], inputs["rho_window"],
    )

    if stream is not None:
        _tasep_core_cuda[blocks, threads_per_block, stream](*kernel_args)
    else:
        _tasep_core_cuda[blocks, threads_per_block](*kernel_args)


# ---------------------------------------------------------------------------
# Host-side: objective_batch_gpu (CMA-ES evaluation)
# ---------------------------------------------------------------------------

def objective_batch_gpu(thetas, base_params, n_runs, S_exp_norm,
                        base_seed=0, device_id=0, stream=None):
    """
    GPU-accelerated batch evaluation of CMA-ES candidates.

    Uses atomic reduction + GPU-side MSE kernel for minimal D2H transfer.

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

    # Transfer dwell profiles to device
    d_dwell = cuda.to_device(inputs["dwell_profiles"], stream=stream)
    d_ribo_dwell = cuda.to_device(inputs["ribo_dwell_profiles"], stream=stream)
    d_rho_dwell = cuda.to_device(inputs["rho_dwell_profiles"], stream=stream)

    # Output: reduced (n_candidates, gene_length) — not (n_total, ...)
    d_netseq = cuda.to_device(
        np.zeros((n_candidates, gene_length), dtype=np.float32), stream=stream)
    # Flux disabled during CMA-ES: tiny dummy allocation
    d_flux_dummy = cuda.to_device(
        np.zeros((1, 1), dtype=np.int32), stream=stream)

    # S_exp_norm on device for MSE kernel
    d_S_exp = cuda.to_device(
        np.asarray(S_exp_norm, dtype=np.float32), stream=stream)
    d_mse = cuda.device_array(n_candidates, dtype=np.float32, stream=stream)

    # Launch TASEP kernel (compute_flux=0)
    _launch_tasep_kernel(inputs, rng_states, d_dwell, d_ribo_dwell, d_rho_dwell,
                         d_netseq, d_flux_dummy, compute_flux=0, stream=stream)

    # Launch MSE kernel (one block per candidate, 256 threads per block)
    mse_args = (d_netseq, d_S_exp, d_mse,
                np.int32(n_runs), np.int32(gene_length), np.int32(n_candidates))
    if stream is not None:
        _compute_mse_cuda[n_candidates, 256, stream](*mse_args)
    else:
        _compute_mse_cuda[n_candidates, 256](*mse_args)

    # Only transfer MSE values to host (~100 bytes)
    h_mse = d_mse.copy_to_host(stream=stream)

    if stream is not None:
        stream.synchronize()
    else:
        cuda.synchronize()

    return h_mse.tolist()


# ---------------------------------------------------------------------------
# Host-side: simulate_with_flux_gpu (post-CMA-ES analysis)
# ---------------------------------------------------------------------------

def simulate_with_flux_gpu(thetas, base_params, n_runs, base_seed=0,
                           device_id=0, stream=None):
    """
    Run simulation with flux computation enabled.

    Use after CMA-ES converges: run a final batch with the optimized D* to
    get both NETseq and flux profiles for analysis/visualization.

    Returns:
        (h_netseq, h_flux) — both (n_candidates, gene_length) arrays
        h_netseq: float32 summed NETseq counts across runs (divide by n_runs for avg)
        h_flux: int32 summed flux counts across runs (divide by n_runs for avg)
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    inputs = _prepare_kernel_inputs(thetas, base_params, n_runs)
    n_total = inputs["n_total"]
    gene_length = inputs["gene_length"]
    n_candidates = inputs["n_candidates"]

    cuda.select_device(device_id)
    rng_states = create_xoroshiro128p_states(n_total, seed=base_seed)

    d_dwell = cuda.to_device(inputs["dwell_profiles"], stream=stream)
    d_ribo_dwell = cuda.to_device(inputs["ribo_dwell_profiles"], stream=stream)
    d_rho_dwell = cuda.to_device(inputs["rho_dwell_profiles"], stream=stream)

    d_netseq = cuda.to_device(
        np.zeros((n_candidates, gene_length), dtype=np.float32), stream=stream)
    d_flux = cuda.to_device(
        np.zeros((n_candidates, gene_length), dtype=np.int32), stream=stream)

    # Launch TASEP kernel (compute_flux=1)
    _launch_tasep_kernel(inputs, rng_states, d_dwell, d_ribo_dwell, d_rho_dwell,
                         d_netseq, d_flux, compute_flux=1, stream=stream)

    h_netseq = d_netseq.copy_to_host(stream=stream)
    h_flux = d_flux.copy_to_host(stream=stream)

    if stream is not None:
        stream.synchronize()
    else:
        cuda.synchronize()

    return h_netseq, h_flux


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

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
