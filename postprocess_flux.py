#!/usr/bin/env python
"""
postprocess_flux.py
===================
Compute normalized flux profiles from saved CMA-ES results.

Supports two backends:

  --mode gpu   Batched GPU kernel (groups genes by gene_length).
  --mode cpu   CPU multithreading via netseq_tasep_fast (no batching needed).
  --mode auto  (default) Use GPU if available, else fall back to CPU.

Usage:
    python postprocess_flux.py --results-dir ecoli --mode cpu --cpu-nt 32
    python postprocess_flux.py --results-dir ecoli --mode gpu --gpu 2
    python postprocess_flux.py --results-dir ecoli --genes insQ,talB --force
"""

import argparse
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def discover_cmaes_results(cmaes_dir: Path, gene_filter: list[str] | None = None) -> list[tuple[str, Path]]:
    """Find completed CMA-ES result .pkl files (exclude _FAILED)."""
    results = []
    for f in sorted(cmaes_dir.glob("*.pkl")):
        if "_FAILED" in f.stem:
            continue
        gene_name = f.stem
        if gene_filter is None or gene_name in gene_filter:
            results.append((gene_name, f))
    return results


def flux_result_path(flux_dir: Path, gene_name: str) -> Path:
    return flux_dir / f"{gene_name}_flux.pkl"


# ---------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------

def _cpu_flux_worker(args):
    """Run one TASEP simulation and return (NETseq_sum, flux)."""
    from netseq_tasep_fast import netseq_tasep_fast
    parameters, seed = args
    result = netseq_tasep_fast(parameters, seed=seed)
    return np.asarray(result["flux"], dtype=np.float64)


def run_flux_cpu(gene_name, cmaes_result, n_runs, dt, n_workers):
    """Compute flux for a single gene using CPU multithreading."""
    theta_best = cmaes_result["theta_best"]
    gene_length = cmaes_result["gene_length"]

    D = np.exp(theta_best)
    D_norm = D / np.mean(D)

    parameters = {
        "KRutLoading": float(cmaes_result["KRutLoading"]),
        "kRiboLoading": float(cmaes_result["kRiboLoading"]),
        "RNAP_dwellTimeProfile": D_norm,
        "dt": dt,
    }

    args_list = [(parameters, i) for i in range(n_runs)]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        flux_results = list(executor.map(_cpu_flux_worker, args_list))

    flux_avg = np.mean(flux_results, axis=0)
    flux_entry = flux_avg[0]
    flux_norm = flux_avg / flux_entry if flux_entry > 0 else flux_avg

    return flux_norm, gene_length


def process_cpu(gene_results, flux_dir, n_runs, dt, n_workers):
    """Process all genes using CPU multithreading."""
    total = len(gene_results)

    print(f"=== Flux Post-Processing (CPU) ===")
    print(f"Flux dir:     {flux_dir}")
    print(f"dt:           {dt}")
    print(f"n_runs:       {n_runs}")
    print(f"CPU threads:  {n_workers}")
    print(f"Genes:        {total}")
    print()

    t_start = time.time()

    for idx, (gene_name, pkl_path) in enumerate(gene_results):
        with open(pkl_path, "rb") as f:
            cmaes_result = pickle.load(f)

        t0 = time.time()
        flux_norm, gene_length = run_flux_cpu(
            gene_name, cmaes_result, n_runs, dt, n_workers)
        elapsed = time.time() - t0

        flux_data = {
            "gene_name": gene_name,
            "gene_length": gene_length,
            "flux_norm": flux_norm,
            "n_runs": n_runs,
            "dt": dt,
        }
        out_path = flux_result_path(flux_dir, gene_name)
        with open(out_path, "wb") as f:
            pickle.dump(flux_data, f)

        rate = (time.time() - t_start) / (idx + 1)
        eta_h = rate * (total - idx - 1) / 3600
        print(f"[CPU] {gene_name} (len={gene_length}): {elapsed:.1f}s | "
              f"{idx+1}/{total} | ETA: {eta_h:.1f}h")

    total_time = time.time() - t_start
    print(f"\nDone. {total} genes in {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Results: {flux_dir}")


# ---------------------------------------------------------------------------
# GPU backend (batched by gene_length)
# ---------------------------------------------------------------------------

def process_gpu(gene_results, flux_dir, n_runs, dt, n_gpus):
    """Process all genes using GPU, batched by gene_length."""
    from netseq_tasep_gpu import simulate_with_flux_gpu
    from numba import cuda as numba_cuda

    # Load all CMA-ES results and group by gene_length
    print("Loading CMA-ES results...")
    gene_data = {}
    length_groups = defaultdict(list)

    for gene_name, pkl_path in gene_results:
        with open(pkl_path, "rb") as f:
            cmaes_result = pickle.load(f)
        gene_data[gene_name] = cmaes_result
        length_groups[cmaes_result["gene_length"]].append(gene_name)

    batches = sorted(length_groups.items(), key=lambda x: len(x[1]), reverse=True)
    total_genes = sum(len(genes) for _, genes in batches)
    total_batches = len(batches)

    batch_sizes = [len(g) for _, g in batches]
    print(f"=== Flux Post-Processing (GPU, Batched) ===")
    print(f"Flux dir:     {flux_dir}")
    print(f"dt:           {dt}")
    print(f"n_runs:       {n_runs}")
    print(f"GPUs:         {n_gpus}")
    print(f"Genes:        {total_genes}")
    print(f"Batches:      {total_batches} (by gene_length)")
    print(f"Batch sizes:  max={max(batch_sizes)}, median={sorted(batch_sizes)[len(batch_sizes)//2]}, "
          f"mean={sum(batch_sizes)/len(batch_sizes):.1f}")
    print()

    t_start = time.time()
    genes_done = 0

    for batch_idx, (gene_length, gene_names) in enumerate(batches):
        batch_size = len(gene_names)
        device_id = batch_idx % n_gpus
        device_label = f"GPU-{device_id}"

        thetas = []
        k_ribo_loadings = []
        for gn in gene_names:
            d = gene_data[gn]
            thetas.append(d["theta_best"])
            k_ribo_loadings.append(float(d["kRiboLoading"]))

        first_gene = gene_data[gene_names[0]]
        base_params = {
            "KRutLoading": float(first_gene["KRutLoading"]),
            "kRiboLoading": 0.0,
            "dt": dt,
        }

        print(f"[{device_label}] Batch {batch_idx+1}/{total_batches}: "
              f"length={gene_length}, {batch_size} genes "
              f"({', '.join(gene_names[:3])}{'...' if batch_size > 3 else ''})")

        numba_cuda.select_device(device_id)
        stream = numba_cuda.stream()

        t0 = time.time()
        h_netseq, h_flux = simulate_with_flux_gpu(
            thetas=thetas,
            base_params=base_params,
            n_runs=n_runs,
            base_seed=0,
            device_id=device_id,
            stream=stream,
            k_ribo_loadings=k_ribo_loadings,
        )
        elapsed = time.time() - t0

        for c_idx, gn in enumerate(gene_names):
            flux_avg = h_flux[c_idx].astype(np.float64) / n_runs
            flux_entry = flux_avg[0]
            flux_norm = flux_avg / flux_entry if flux_entry > 0 else flux_avg

            flux_data = {
                "gene_name": gn,
                "gene_length": gene_length,
                "flux_norm": flux_norm,
                "n_runs": n_runs,
                "dt": dt,
            }
            out_path = flux_result_path(flux_dir, gn)
            with open(out_path, "wb") as f:
                pickle.dump(flux_data, f)

        genes_done += batch_size
        rate = (time.time() - t_start) / genes_done
        eta_h = rate * (total_genes - genes_done) / 3600

        print(f"[{device_label}] Batch done: {elapsed:.1f}s ({elapsed/batch_size:.2f}s/gene) | "
              f"{genes_done}/{total_genes} genes | ETA: {eta_h:.1f}h")

    total_time = time.time() - t_start
    print(f"\nDone. {total_genes} genes in {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Results: {flux_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process flux from CMA-ES results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Results directory containing cmaes/ subdirectory (default: results/)")
    parser.add_argument("--mode", choices=["gpu", "cpu", "auto"], default="auto",
                        help="Backend: gpu (batched), cpu (multithreaded), or auto (default: auto)")
    parser.add_argument("--gpu", type=int, default=1,
                        help="Number of GPUs to use in gpu mode (default: 1)")
    parser.add_argument("--cpu-nt", type=int, default=0,
                        help="CPU threads for cpu mode (default: all cores)")
    parser.add_argument("--genes", type=str, default=None,
                        help="Comma-separated gene names to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing of already-completed genes")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override flux output directory (default: <results-dir>/flux). "
                             "Use e.g. <results-dir>/flux_patched to write a parallel set of "
                             "outputs without overwriting the existing flux pickles.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cmaes_dir = results_dir / "cmaes"
    flux_dir = Path(args.output_dir) if args.output_dir else results_dir / "flux"
    flux_dir.mkdir(parents=True, exist_ok=True)

    if not cmaes_dir.exists():
        print(f"No cmaes/ directory found at {cmaes_dir}")
        return

    # Discover genes
    gene_filter = None
    if args.genes:
        gene_filter = [g.strip() for g in args.genes.split(",")]
    gene_results = discover_cmaes_results(cmaes_dir, gene_filter)

    if not gene_results:
        print("No CMA-ES results found.")
        return

    # Skip already-processed genes
    if not args.force:
        already_done = set()
        for gene_name, _ in gene_results:
            if flux_result_path(flux_dir, gene_name).exists():
                already_done.add(gene_name)
        if already_done:
            print(f"Skipping {len(already_done)} already-processed genes")
            gene_results = [(g, p) for g, p in gene_results if g not in already_done]

    if not gene_results:
        print("All genes already processed. Use --force to re-run.")
        return

    n_runs = 200
    dt = 0.1

    # Resolve mode
    mode = args.mode
    if mode == "auto":
        try:
            from netseq_tasep_gpu import cuda_is_available
            if cuda_is_available():
                mode = "gpu"
            else:
                mode = "cpu"
        except ImportError:
            mode = "cpu"
        print(f"Auto-detected mode: {mode}")

    if mode == "gpu":
        from netseq_tasep_gpu import cuda_is_available, gpu_count
        if not cuda_is_available():
            print("CUDA not available. Falling back to CPU mode.")
            mode = "cpu"
        else:
            n_gpus = min(args.gpu, gpu_count())
            process_gpu(gene_results, flux_dir, n_runs, dt, n_gpus)
            return

    # CPU mode
    n_workers = args.cpu_nt if args.cpu_nt > 0 else (os.cpu_count() or 1)
    process_cpu(gene_results, flux_dir, n_runs, dt, n_workers)


if __name__ == "__main__":
    main()
