from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _exp_rnd_scalar(rng: np.random.Generator, mean: float) -> float:
    if mean <= 0 or not math.isfinite(mean):
        return math.inf
    return float(rng.exponential(scale=mean))


def _exp_rnd_array(rng: np.random.Generator, means: np.ndarray) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    means = np.where(means > 0, means, 0.0)
    return rng.exponential(scale=means)


def netseq_tasep_function(
    input_parameters: dict | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = np.random.default_rng() if rng is None else rng

    parameters = {
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
    if input_parameters:
        for key, value in input_parameters.items():
            parameters[key] = value

    parameters["rutSpeed"] = 5 * parameters["ribospeed"]
    dwell_profile = np.asarray(parameters["RNAP_dwellTimeProfile"], dtype=float).reshape(-1)
    parameters["geneLength"] = len(dwell_profile)

    rnap_speed = float(parameters["RNAPSpeed"])
    ribo_speed = float(parameters["ribospeed"])
    gene_length = int(parameters["geneLength"])
    rnap_width = 35
    dx = 1
    dt = 0.1
    simtime = int(parameters["simtime"])
    glutime = float(parameters["glutime"])
    k_loading = float(parameters["kLoading"])
    k_ribo_loading = float(parameters["kRiboLoading"])
    bool_rnap_ribo_coupling = 1

    bases_rnap = int(math.ceil(rnap_speed * 10 * dt))
    bases_ribo = int(math.ceil(ribo_speed * 10 * dt))
    rho_window = max(1, int(round(ribo_speed * 5)))

    rnap_exit_times: list[np.ndarray] = []
    ribo_exit_times: list[np.ndarray] = []
    rnap_locs: list[int] = []
    ribo_locs: list[int] = []
    rho_locs: list[int] = []
    rnap_ribo_coupling: list[int] = []
    ribo_loadt: list[float] = []
    rut_loadt: list[list[float]] = []

    rut_sites = [int(round(500 * gene_length / 3075))]
    rut_speed = float(parameters["rutSpeed"])
    min_rho_load_rna = 80 - 30
    specific_dwelltime_rho = np.zeros(gene_length + 1)
    specific_dwelltime_rho[1:] = dx / rut_speed
    pt_percent = 0.0
    pt_model = 2
    if pt_model in (1, 2):
        pt_percent = float(parameters["KRutLoading"])

    avg_dwelltime = dx / rnap_speed
    ribo_avg_dwelltime = dx / ribo_speed
    loadt = _exp_rnd_scalar(rng, 1 / k_loading) if k_loading > 0 else math.inf

    specific_dwelltime1 = np.zeros(gene_length + 1)
    specific_dwelltime1[1:] = avg_dwelltime * dwell_profile
    ribo_specific_dwelltime1 = np.zeros(gene_length + 1)
    ribo_specific_dwelltime1[1:] = ribo_avg_dwelltime

    n_steps = int(simtime / dt)
    for step in range(n_steps + 1):
        t = step * dt

        if (
            loadt <= t
            and t < glutime
            and rnap_locs
            and (rnap_locs[-1] - rnap_width) <= 0
        ):
            loadt = t + _exp_rnd_scalar(rng, 1 / k_loading) if k_loading > 0 else math.inf

        if loadt <= t and t < glutime and (not rnap_locs or (rnap_locs[-1] - rnap_width) >= 0):
            rnap_locs.append(1)
            rnap_exit_times.append(np.zeros(gene_length + 1))
            rnap_ribo_coupling.append(0)
            loadt = t + _exp_rnd_scalar(rng, 1 / k_loading) if k_loading > 0 else math.inf
            ribo_loadt.append(
                t + _exp_rnd_scalar(rng, 1 / k_ribo_loading) if k_ribo_loading > 0 else math.inf
            )
            ribo_locs.append(0)
            rho_locs.append(0)
            ribo_exit_times.append(np.zeros(gene_length + 1))
            rut_loadt.append([simtime + 1] * len(rut_sites))

        for idx in range(len(rnap_locs)):
            current_loc = rnap_locs[idx]
            if current_loc <= gene_length:
                if t < glutime:
                    if current_loc + bases_rnap <= gene_length:
                        window = specific_dwelltime1[current_loc : current_loc + bases_rnap + 1]
                    else:
                        window = specific_dwelltime1[current_loc : gene_length + 1]
                    temp_exit_times = t + np.cumsum(_exp_rnd_array(rng, window))
                else:
                    temp_exit_times = np.array([], dtype=float)

                temp_rnap_exit = temp_exit_times[
                    (temp_exit_times >= t) & (temp_exit_times <= t + dt)
                ]
                advance = len(temp_rnap_exit)

                if idx > 0:
                    prev_loc = rnap_locs[idx - 1]
                    if prev_loc == gene_length + 10:
                        j = 1
                        while j <= idx and rnap_locs[idx - j] == gene_length + 10 and (idx - j) > 0:
                            j += 1
                        if j == idx + 1 or (idx - j) < 0:
                            prev_loc = gene_length + 1
                        else:
                            prev_loc = rnap_locs[idx - j]

                    overlap = (current_loc + advance - 1) - prev_loc + rnap_width
                    if prev_loc >= gene_length:
                        overlap = 0

                    if overlap <= 0 and advance > 0:
                        rnap_exit_times[idx][current_loc : current_loc + advance] = temp_rnap_exit
                        rnap_locs[idx] = current_loc + advance
                    elif overlap > 0:
                        advance2 = advance - overlap
                        if advance2 > 0:
                            rnap_exit_times[idx][current_loc : current_loc + advance2] = temp_rnap_exit[
                                :advance2
                            ]
                            rnap_locs[idx] = current_loc + advance2
                else:
                    if advance > 0:
                        rnap_exit_times[idx][current_loc : current_loc + advance] = temp_rnap_exit
                        rnap_locs[idx] = current_loc + advance

            if ribo_loadt[idx] <= t and rnap_locs[idx] >= 30:
                ribo_loadt[idx] = simtime
                ribo_locs[idx] = 1
                ribo_exit_times[idx][1] = t

        for idx in range(len(rnap_locs)):
            if pt_model == 2 and rnap_locs[idx] < gene_length:
                ptrna_size = rnap_locs[idx] - rnap_width - 30
                ptrna_size -= ribo_locs[idx]
                if (
                    ptrna_size > min_rho_load_rna
                    and 100 * dt * rng.random() <= pt_percent * ptrna_size / gene_length
                ):
                    temp_rho_loc = rnap_locs[idx] - rnap_width - int(
                        math.floor(rng.random() * ptrna_size)
                    )
                    if temp_rho_loc > rho_locs[idx]:
                        rho_locs[idx] = temp_rho_loc

            for rs_idx, rut_site in enumerate(rut_sites):
                if (
                    pt_model == 1
                    and rnap_exit_times[idx][rut_site] < t + dt
                    and rnap_exit_times[idx][rut_site] > t
                    and 100 * rng.random() <= pt_percent
                ):
                    rnap_locs[idx] = gene_length + 10

                if (
                    pt_model == 0
                    and ribo_locs[idx] <= rut_site
                    and rnap_exit_times[idx][rut_site] < t + dt
                    and rnap_exit_times[idx][rut_site] > t
                ):
                    rut_loadt[idx][rs_idx] = t + _exp_rnd_scalar(rng, 1 / parameters["KRutLoading"])

                if (
                    t > rut_loadt[idx][rs_idx]
                    and rho_locs[idx] < rut_site
                    and ribo_locs[idx] < rut_site
                    and rnap_locs[idx] > rut_site
                    and rnap_ribo_coupling[idx] == 0
                    and rnap_locs[idx] < gene_length + 1
                ):
                    rho_locs[idx] = rut_site

            if rnap_locs[idx] == gene_length + 10:
                rho_locs[idx] = gene_length + 9

            if 0 < rho_locs[idx] < gene_length:
                if rho_locs[idx] + rho_window <= gene_length:
                    window = specific_dwelltime_rho[
                        rho_locs[idx] : rho_locs[idx] + rho_window + 1
                    ]
                else:
                    window = specific_dwelltime_rho[rho_locs[idx] : gene_length + 1]
                temp_exit = t + np.cumsum(_exp_rnd_array(rng, window))
                temp_rho = temp_exit[(temp_exit >= t) & (temp_exit <= t + dt)]
                rho_locs[idx] += len(temp_rho)

            if rho_locs[idx] >= rnap_locs[idx]:
                rnap_locs[idx] = gene_length + 10
                rho_locs[idx] = gene_length + 9

        for idx in range(len(ribo_locs)):
            if 0 < ribo_locs[idx] <= gene_length:
                if ribo_locs[idx] + bases_ribo <= gene_length:
                    window = ribo_specific_dwelltime1[
                        ribo_locs[idx] : ribo_locs[idx] + bases_ribo + 1
                    ]
                else:
                    window = ribo_specific_dwelltime1[ribo_locs[idx] : gene_length + 1]
                temp_exit2 = t + np.cumsum(_exp_rnd_array(rng, window))
                temp_ribo_exit = temp_exit2[(temp_exit2 >= t) & (temp_exit2 <= t + dt)]

                if rnap_ribo_coupling[idx] == 1 and ribo_locs[idx] <= gene_length - rnap_width:
                    ribo_exit_times[idx][ribo_locs[idx] : gene_length - rnap_width + 1] = (
                        rnap_exit_times[idx][ribo_locs[idx] + rnap_width : gene_length + 1]
                    )
                    ribo_locs[idx] = rnap_locs[idx] - rnap_width
                elif (
                    rnap_ribo_coupling[idx] == 1
                    and ribo_locs[idx] > gene_length - 30
                    and ribo_locs[idx] < gene_length + 1
                ):
                    window = ribo_specific_dwelltime1[ribo_locs[idx] : gene_length + 1]
                    ribo_exit_times[idx][ribo_locs[idx] : gene_length + 1] = t + np.cumsum(
                        _exp_rnd_array(rng, window)
                    )
                    ribo_locs[idx] = gene_length + 1
                elif rnap_locs[idx] == gene_length + 10:
                    last_idx = int(np.sum(rnap_exit_times[idx] > 0))
                    if last_idx < ribo_locs[idx]:
                        last_idx = ribo_locs[idx]
                    window = ribo_specific_dwelltime1[ribo_locs[idx] : last_idx + 1]
                    ribo_exit_times[idx][ribo_locs[idx] : last_idx + 1] = t + np.cumsum(
                        _exp_rnd_array(rng, window)
                    )
                    idx2 = int(np.count_nonzero(rnap_exit_times[idx])) + 1
                    if idx2 <= gene_length:
                        ribo_exit_times[idx][idx2 : gene_length + 1] = 0
                    ribo_locs[idx] = gene_length + 10
                else:
                    overlap = (ribo_locs[idx] + len(temp_ribo_exit) - 1) - rnap_locs[idx] + rnap_width
                    if rnap_locs[idx] == gene_length + 1:
                        overlap = 0
                    if overlap > 0:
                        if rnap_locs[idx] <= gene_length and bool_rnap_ribo_coupling == 1:
                            rnap_ribo_coupling[idx] = 1
                        advance = len(temp_ribo_exit) - overlap
                        if advance > 0:
                            ribo_exit_times[idx][
                                ribo_locs[idx] : ribo_locs[idx] + advance
                            ] = temp_ribo_exit[:advance]
                            ribo_locs[idx] = rnap_locs[idx] - rnap_width
                    else:
                        if len(temp_ribo_exit) > 0:
                            ribo_exit_times[idx][
                                ribo_locs[idx] : ribo_locs[idx] + len(temp_ribo_exit)
                            ] = temp_ribo_exit
                            ribo_locs[idx] += len(temp_ribo_exit)

    netseq = np.zeros((gene_length + 1, simtime + 1), dtype=float)
    if rnap_exit_times:
        rnap_exit_matrix = np.stack(rnap_exit_times, axis=1)
        max_per_rnap = np.max(rnap_exit_matrix, axis=0)
        for t in range(1, simtime + 1):
            temp = np.sum((rnap_exit_matrix <= t) & (rnap_exit_matrix > 0), axis=0)
            mask = (temp > 0) & (temp < gene_length) & (max_per_rnap > t)
            temp = temp[mask]
            if temp.size > 0:
                counts = np.bincount(temp.astype(int), minlength=gene_length + 1)
                netseq[:, t] = counts[: gene_length + 1]

    return {
        "parameters": parameters,
        "NETseq": netseq[1:, 1:],
    }


def run_netseq_simulations(gene: str, n_runs: int, seed: int | None = None) -> tuple[np.ndarray, Path]:
    rng = np.random.default_rng(seed)
    script_dir = Path(__file__).resolve().parent
    te_path = script_dir / "Ecoli_gene_TE.csv"
    gene_profile_path = script_dir / "NETSEQ_gene" / f"NETSEQ_{gene}.csv"

    if not te_path.exists():
        raise FileNotFoundError(f"Missing {te_path}")
    if not gene_profile_path.exists():
        raise FileNotFoundError(f"Missing {gene_profile_path}")

    te_df = pd.read_csv(te_path)
    normalized = {
        "".join(ch for ch in col.lower() if ch.isalnum()): col for col in te_df.columns
    }
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
        kribo = float(te_values.iloc[0]) / 5

    dwell_profile = np.loadtxt(gene_profile_path, delimiter=",")
    dwell_profile = dwell_profile / np.mean(dwell_profile)

    parameters = {
        "KRutLoading": 0.13,
        "RNAP_dwellTimeProfile": dwell_profile,
        "kRiboLoading": kribo,
    }

    final_output = netseq_tasep_function(parameters, rng)
    netseq_total = final_output["NETseq"].astype(float)
    for _ in range(n_runs):
        output = netseq_tasep_function(parameters, rng)
        netseq_total += output["NETseq"]
    netseq_total = netseq_total / n_runs

    cols = [i - 1 for i in range(200, min(1500, netseq_total.shape[1]) + 1, 200)]
    netseq_sum = np.sum(netseq_total[:, cols], axis=1) if cols else np.zeros(netseq_total.shape[0])

    out_path = script_dir / f"NETSEQ_{gene}_NETseqSum.png"
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(netseq_sum) + 1), netseq_sum)
    plt.xlabel("Position (bp)")
    plt.ylabel("NETseq sum")
    plt.title(f"NETseq sum for {gene}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return netseq_sum, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NETSEQ TASEP simulation and plot results.")
    parser.add_argument("--gene", default="insQ", help="Gene name (e.g., insQ).")
    parser.add_argument("--n-runs", type=int, default=300, help="Number of runs to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()

    netseq_sum, out_path = run_netseq_simulations(args.gene, args.n_runs, args.seed)
    print(f"Saved plot: {out_path}")
    print(f"NETseq sum length: {len(netseq_sum)}")


if __name__ == "__main__":
    main()
