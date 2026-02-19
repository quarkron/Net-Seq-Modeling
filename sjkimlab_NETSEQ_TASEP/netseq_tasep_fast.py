from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


# Legacy reference from OpenSpec task artifact; runtime caps are estimated per run.
MAX_RNAP = 90
SNAPSHOTS = [200, 400, 600, 800, 1000, 1200, 1400]
ACTIVE_CAP_SAFETY_MARGIN = 1.25
HISTORY_CAP_BUFFER_FACTOR = 2.0
HISTORY_CAP_BUFFER_ADD = 32
HISTORY_CAP_MIN_GAP = 8


_EXECUTOR_CACHE: dict[int, ProcessPoolExecutor] = {}
_EXECUTOR_TOKEN: dict[int, tuple[int, float, float, float, float]] = {}
_WORKER_PARAMETERS: dict | None = None


@njit(cache=False)
def _sample_first_passage(
    dwell_profile: np.ndarray,
    start_pos: int,
    end_pos: int,
    t_now: float,
    dt: float,
    exit_buffer: np.ndarray,
) -> int:
    """Simulate cumulative exponential exits from start_pos to end_pos."""
    next_time = t_now
    count = 0
    for pos in range(start_pos, end_pos + 1):
        scale = dwell_profile[pos]
        if scale > 0.0:
            next_time += np.random.exponential(scale)
        if next_time > t_now + dt:
            break
        exit_buffer[count] = next_time
        count += 1
    return count


@njit(cache=False)
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
    Numba-compiled core dynamics.

    Returns rnap_exit_matrix and n_rnap where n_rnap is total loaded RNAP count.
    """
    np.random.seed(seed)

    term_rnap = gene_length + 10
    term_rho = gene_length + 9
    finished_rnap = gene_length + 1
    big_t = simtime + 1.0

    rnap_exit_matrix = np.zeros((history_capacity, gene_length + 1), dtype=np.float64)
    ribo_exit_matrix = np.zeros((history_capacity, gene_length + 1), dtype=np.float64)

    rnap_locs = np.zeros(history_capacity, dtype=np.int64)
    ribo_locs = np.zeros(history_capacity, dtype=np.int64)
    rho_locs = np.zeros(history_capacity, dtype=np.int64)

    rnap_ribo_coupling = np.zeros(history_capacity, dtype=np.int64)
    ribo_loadt = np.full(history_capacity, big_t, dtype=np.float64)

    if k_loading > 0.0:
        loadt = np.random.exponential(1.0 / k_loading)
    else:
        loadt = big_t

    n_rnap = 0
    n_steps = int(simtime / dt)

    rnap_exit_buffer = np.empty(bases_rnap + 2, dtype=np.float64)
    ribo_exit_buffer = np.empty(bases_ribo + 2, dtype=np.float64)
    rho_exit_buffer = np.empty(rho_window + 2, dtype=np.float64)

    for step in range(n_steps + 1):
        t_now = step * dt

        # RNAP loading while promoter is active.
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

        if (
            loadt <= t_now
            and t_now < glutime
            and (n_rnap == 0 or (rnap_locs[n_rnap - 1] - rnap_width) >= 0)
        ):
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
            rnap_locs[idx] = 1
            rnap_ribo_coupling[idx] = 0
            ribo_locs[idx] = 0
            rho_locs[idx] = 0
            if k_loading > 0.0:
                loadt = t_now + np.random.exponential(1.0 / k_loading)
            else:
                loadt = big_t
            if k_ribo_loading > 0.0:
                ribo_loadt[idx] = t_now + np.random.exponential(1.0 / k_ribo_loading)
            else:
                ribo_loadt[idx] = big_t

        # RNAP elongation and ribosome loading.
        for idx in range(n_rnap):
            curr_loc = rnap_locs[idx]
            if curr_loc <= gene_length:
                end_loc = curr_loc + bases_rnap
                if end_loc > gene_length:
                    end_loc = gene_length

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
                    if idx > 0:
                        prev_loc = rnap_locs[idx - 1]
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

                    if allowed_advance > 0:
                        for k in range(allowed_advance):
                            rnap_exit_matrix[idx, curr_loc + k] = rnap_exit_buffer[k]
                        rnap_locs[idx] = curr_loc + allowed_advance

            if ribo_loadt[idx] <= t_now and rnap_locs[idx] >= 30:
                ribo_loadt[idx] = big_t
                ribo_locs[idx] = 1
                ribo_exit_matrix[idx, 1] = t_now

        # PT_Model=2 Rho loading + Rho chase.
        for idx in range(n_rnap):
            if rnap_locs[idx] < gene_length:
                ptrna_size = rnap_locs[idx] - rnap_width - 30 - ribo_locs[idx]
                if ptrna_size > min_rho_load_rna:
                    threshold = pt_percent * ptrna_size / gene_length
                    if 100.0 * dt * np.random.random() <= threshold:
                        temp_rho_loc = rnap_locs[idx] - rnap_width - int(
                            np.floor(np.random.random() * ptrna_size)
                        )
                        if temp_rho_loc > rho_locs[idx]:
                            rho_locs[idx] = temp_rho_loc

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

            if rho_locs[idx] >= rnap_locs[idx]:
                rnap_locs[idx] = term_rnap
                rho_locs[idx] = term_rho

        # Ribosome elongation and RNAP-ribosome coupling.
        for idx in range(n_rnap):
            ribo_loc = ribo_locs[idx]
            if ribo_loc > 0 and ribo_loc <= gene_length:
                if rnap_ribo_coupling[idx] == 1 and ribo_loc <= gene_length - rnap_width:
                    for pos in range(ribo_loc, gene_length - rnap_width + 1):
                        ribo_exit_matrix[idx, pos] = rnap_exit_matrix[idx, pos + rnap_width]
                    ribo_locs[idx] = rnap_locs[idx] - rnap_width
                elif (
                    rnap_ribo_coupling[idx] == 1
                    and ribo_loc > gene_length - 30
                    and ribo_loc < gene_length + 1
                ):
                    finish_t = t_now
                    for pos in range(ribo_loc, gene_length + 1):
                        dwell = ribo_specific_dwelltime1[pos]
                        if dwell > 0.0:
                            finish_t += np.random.exponential(dwell)
                        ribo_exit_matrix[idx, pos] = finish_t
                    ribo_locs[idx] = finished_rnap
                elif rnap_locs[idx] == term_rnap:
                    last_idx = 0
                    for pos in range(1, gene_length + 1):
                        if rnap_exit_matrix[idx, pos] > 0.0:
                            last_idx = pos
                    if last_idx < ribo_loc:
                        last_idx = ribo_loc
                    finish_t = t_now
                    for pos in range(ribo_loc, last_idx + 1):
                        dwell = ribo_specific_dwelltime1[pos]
                        if dwell > 0.0:
                            finish_t += np.random.exponential(dwell)
                        ribo_exit_matrix[idx, pos] = finish_t
                    for pos in range(last_idx + 1, gene_length + 1):
                        ribo_exit_matrix[idx, pos] = 0.0
                    ribo_locs[idx] = term_rnap
                else:
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
                    overlap = (ribo_loc + ribo_advance - 1) - rnap_locs[idx] + rnap_width
                    if rnap_locs[idx] == finished_rnap:
                        overlap = 0
                    if overlap > 0:
                        if rnap_locs[idx] <= gene_length:
                            rnap_ribo_coupling[idx] = 1
                        allowed_advance = ribo_advance - overlap
                        if allowed_advance > 0:
                            for k in range(allowed_advance):
                                ribo_exit_matrix[idx, ribo_loc + k] = ribo_exit_buffer[k]
                            ribo_locs[idx] = rnap_locs[idx] - rnap_width
                    elif ribo_advance > 0:
                        for k in range(ribo_advance):
                            ribo_exit_matrix[idx, ribo_loc + k] = ribo_exit_buffer[k]
                        ribo_locs[idx] = ribo_loc + ribo_advance

    return rnap_exit_matrix, n_rnap


def _compute_netseq_sum(
    rnap_exit_matrix: np.ndarray,
    n_rnap: int,
    gene_length: int,
    snapshots: np.ndarray,
) -> np.ndarray:
    """Compute snapshot-summed NET-seq using vectorized position counts."""
    netseq_sum = np.zeros(gene_length + 1, dtype=float)
    if n_rnap == 0:
        return netseq_sum[1:]

    exits = rnap_exit_matrix[:n_rnap, : gene_length + 1]
    max_per_rnap = np.max(exits, axis=1)

    for snapshot in snapshots:
        temp = np.sum((exits <= snapshot) & (exits > 0.0), axis=1)
        mask = (temp > 0) & (temp < gene_length) & (max_per_rnap > snapshot)
        if np.any(mask):
            netseq_sum += np.bincount(temp[mask].astype(np.int64), minlength=gene_length + 1)[
                : gene_length + 1
            ]

    return netseq_sum[1:]


def _default_parameters() -> dict:
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
    """Estimate biologically grounded active/history capacities for preallocation."""
    packing_cap = max(1, int(math.ceil(gene_length / float(rnap_width))))
    active_cap = max(1, int(math.ceil(packing_cap * ACTIVE_CAP_SAFETY_MARGIN)))
    expected_loads = max(1.0, k_loading * simtime)
    history_cap = max(
        active_cap + HISTORY_CAP_MIN_GAP,
        int(math.ceil(expected_loads * HISTORY_CAP_BUFFER_FACTOR + HISTORY_CAP_BUFFER_ADD)),
    )
    return active_cap, history_cap


def netseq_tasep_fast(input_parameters: dict | None = None, seed: int = 0) -> dict:
    """
    Run one fast TASEP simulation and return RL-ready snapshot sum.

    Returns {"parameters": ..., "NETseq_sum": np.ndarray}.
    """
    parameters = _default_parameters()
    if input_parameters:
        parameters.update(input_parameters)

    dwell_profile = np.asarray(parameters["RNAP_dwellTimeProfile"], dtype=float).reshape(-1)
    gene_length = int(dwell_profile.size)
    parameters["geneLength"] = gene_length
    parameters["rutSpeed"] = 5 * float(parameters["ribospeed"])

    rnap_speed = float(parameters["RNAPSpeed"])
    ribo_speed = float(parameters["ribospeed"])
    simtime = int(parameters["simtime"])
    glutime = float(parameters["glutime"])
    k_loading = float(parameters["kLoading"])
    k_ribo_loading = float(parameters["kRiboLoading"])
    pt_percent = float(parameters["KRutLoading"])

    rnap_width = 35
    dx = 1.0
    dt = 0.1
    min_rho_load_rna = 80 - 30

    bases_rnap = int(math.ceil(rnap_speed * 10.0 * dt))
    bases_ribo = int(math.ceil(ribo_speed * 10.0 * dt))
    rho_window = max(1, int(round(ribo_speed * 5.0)))

    specific_dwelltime1 = np.zeros(gene_length + 1, dtype=np.float64)
    specific_dwelltime1[1:] = (dx / rnap_speed) * dwell_profile

    ribo_specific_dwelltime1 = np.zeros(gene_length + 1, dtype=np.float64)
    ribo_specific_dwelltime1[1:] = dx / ribo_speed

    specific_dwelltime_rho = np.zeros(gene_length + 1, dtype=np.float64)
    rut_speed = float(parameters["rutSpeed"])
    if rut_speed > 0.0:
        specific_dwelltime_rho[1:] = dx / rut_speed

    active_cap, history_cap = _estimate_caps(
        gene_length=gene_length,
        rnap_width=rnap_width,
        k_loading=k_loading,
        simtime=simtime,
    )
    parameters["activeCap"] = int(active_cap)
    parameters["historyCap"] = int(history_cap)

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

    netseq_sum = _compute_netseq_sum(
        rnap_exit_matrix=rnap_exit_matrix,
        n_rnap=n_rnap,
        gene_length=gene_length,
        snapshots=np.asarray(SNAPSHOTS, dtype=np.int64),
    )
    return {"parameters": parameters, "NETseq_sum": netseq_sum}


def _load_gene_parameters(gene: str) -> dict:
    script_dir = Path(__file__).resolve().parent
    te_path = script_dir / "Ecoli_gene_TE.csv"
    gene_profile_path = script_dir / "NETSEQ_gene" / f"NETSEQ_{gene}.csv"

    if not te_path.exists():
        raise FileNotFoundError(f"Missing {te_path}")
    if not gene_profile_path.exists():
        raise FileNotFoundError(f"Missing {gene_profile_path}")

    te_df = pd.read_csv(te_path)
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
        kribo = 0.0
    else:
        kribo = float(te_values.iloc[0]) / 5.0

    dwell_profile = np.loadtxt(gene_profile_path, delimiter=",")
    dwell_profile = dwell_profile / np.mean(dwell_profile)

    return {
        "KRutLoading": 0.13,
        "RNAP_dwellTimeProfile": dwell_profile,
        "kRiboLoading": kribo,
    }


def _worker(args: tuple[dict, int]) -> np.ndarray:
    parameters, seed = args
    result = netseq_tasep_fast(parameters, seed=seed)
    return np.asarray(result["NETseq_sum"], dtype=float)


def _parameters_token(parameters: dict) -> tuple[int, float, float, float, float]:
    dwell = np.asarray(parameters["RNAP_dwellTimeProfile"], dtype=float)
    return (
        int(dwell.size),
        float(parameters.get("kRiboLoading", 0.0)),
        float(parameters.get("KRutLoading", 0.0)),
        float(np.sum(dwell)),
        float(np.sum(dwell * dwell)),
    )


def _init_worker_parameters(parameters: dict) -> None:
    global _WORKER_PARAMETERS
    _WORKER_PARAMETERS = parameters


def _worker_seed(seed: int) -> np.ndarray:
    if _WORKER_PARAMETERS is None:
        raise ValueError("worker parameters are not initialized")
    result = netseq_tasep_fast(_WORKER_PARAMETERS, seed=int(seed))
    return np.asarray(result["NETseq_sum"], dtype=float)


def _worker_seed_batch(seeds: list[int]) -> np.ndarray:
    netseq_total = None
    for seed in seeds:
        output = _worker_seed(int(seed))
        if netseq_total is None:
            netseq_total = output
        else:
            netseq_total += output
    if netseq_total is None:
        raise ValueError("worker seed batch received no seeds")
    return netseq_total


def _worker_batch(args: tuple[dict, list[int]]) -> np.ndarray:
    parameters, seeds = args
    netseq_total = None
    for seed in seeds:
        output = _worker((parameters, seed))
        if netseq_total is None:
            netseq_total = output
        else:
            netseq_total += output
    if netseq_total is None:
        raise ValueError("worker batch received no seeds")
    return netseq_total


def run_netseq_simulations_fast(
    gene: str,
    n_runs: int,
    seed: int | None = None,
    n_workers: int | None = None,
) -> np.ndarray:
    """
    Average exactly n_runs fast simulations over seeds (seed + i).
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    parameters = _load_gene_parameters(gene)
    base_seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    seeds = [base_seed + i for i in range(n_runs)]

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        outputs = [_worker((parameters, s)) for s in seeds]
        netseq_total = np.zeros_like(outputs[0], dtype=float)
        for output in outputs:
            netseq_total += output
    else:
        n_batches = min(n_workers, n_runs)
        seed_chunks = [chunk.tolist() for chunk in np.array_split(np.asarray(seeds), n_batches)]
        worker_args = [chunk for chunk in seed_chunks if len(chunk) > 0]
        params_token = _parameters_token(parameters)
        executor = _EXECUTOR_CACHE.get(n_workers)
        token = _EXECUTOR_TOKEN.get(n_workers)
        if executor is None or token != params_token:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            executor = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker_parameters,
                initargs=(parameters,),
            )
            _EXECUTOR_CACHE[n_workers] = executor
            _EXECUTOR_TOKEN[n_workers] = params_token
        try:
            outputs = list(executor.map(_worker_seed_batch, worker_args))
        except BrokenProcessPool:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker_parameters,
                initargs=(parameters,),
            )
            _EXECUTOR_CACHE[n_workers] = executor
            _EXECUTOR_TOKEN[n_workers] = params_token
            outputs = list(executor.map(_worker_seed_batch, worker_args))
        netseq_total = np.zeros_like(outputs[0], dtype=float)
        for output in outputs:
            netseq_total += output

    return netseq_total / float(n_runs)
