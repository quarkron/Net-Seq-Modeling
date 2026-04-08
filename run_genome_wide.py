#!/usr/bin/env python
"""
run_genome_wide.py
==================
Genome-wide CMA-ES deconvolution runner.

Uses multiprocessing (1 process per worker) to avoid GIL contention between
GPU workers and CPU workers.

Usage:
    python run_genome_wide.py --output results --gpu 2
    python run_genome_wide.py --output results --gpu 2 --cpu-nt 16
    python run_genome_wide.py --output results --gpu 2 --genes insQ,talB,aceA
"""

import argparse
import csv
import math
import multiprocessing
import os
import pickle
import signal
import sys
import time
import traceback
from multiprocessing import Process, Queue as MPQueue, Value
from pathlib import Path

import numpy as np


def discover_genes(netseq_dir: Path, gene_filter: list[str] | None = None) -> list[str]:
    """Scan NETSEQ_gene/*.csv and return sorted list of gene names."""
    genes = []
    for f in sorted(netseq_dir.glob("NETSEQ_*.csv")):
        name = f.stem.replace("NETSEQ_", "")
        if gene_filter is None or name in gene_filter:
            genes.append(name)
    return genes


def gene_result_path(output_dir: Path, gene_name: str) -> Path:
    return output_dir / f"{gene_name}.pkl"


def gene_failed_path(output_dir: Path, gene_name: str) -> Path:
    return output_dir / f"{gene_name}_FAILED.pkl"


def save_result(path: Path, result: dict):
    with open(path, "wb") as f:
        pickle.dump(result, f)


def append_summary_row(csv_path: Path, row: dict):
    """Append one row to the summary CSV (create with header if new)."""
    fieldnames = [
        "gene", "gene_length", "KRutLoading", "kRiboLoading",
        "final_mse", "final_rms", "generations", "early_stopped", "wall_time",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _gene_length_from_csv(netseq_dir: Path, gene_name: str) -> int:
    """Get gene length from CSV file."""
    path = netseq_dir / f"NETSEQ_{gene_name}.csv"
    try:
        return len(np.loadtxt(path, delimiter=","))
    except Exception:
        return 0


def worker_process(gene_queue: MPQueue, result_queue: MPQueue, output_dir: Path,
                   config_dict: dict, device_id: int, netseq_dir: Path,
                   gen_counter=None):
    """Worker process: pull genes from queue, process them, send results back."""
    from cmaes_multifidelity import CMAESConfig, NRunsSchedule, EarlyStopping, run_cmaes_for_gene

    device_label = f"GPU-{device_id}" if device_id >= 0 else "CPU"

    # Reconstruct config in this process
    config = CMAESConfig(
        sigma0=config_dict["sigma0"],
        max_generations=config_dict["max_generations"],
        use_gpu=config_dict["use_gpu"],
        device_id=device_id,
        display_every=0,
        cpu_nt=config_dict.get("cpu_nt"),
        dt=config_dict["dt"],
        gen_counter=gen_counter,
        n_runs_schedule=NRunsSchedule(
            n_runs_low=config_dict["n_runs"],
            n_runs_med=config_dict["n_runs"],
            n_runs_high=config_dict["n_runs"],
        ),
        early_stopping=EarlyStopping(
            window=config_dict["early_stop_window"],
            rel_threshold=config_dict["early_stop_threshold"],
        ),
    )

    # Create CUDA stream if GPU worker
    stream = None
    if config.use_gpu and device_id >= 0:
        from numba import cuda
        cuda.select_device(device_id)
        stream = cuda.stream()

    while True:
        item = gene_queue.get()
        if item is None:  # poison pill
            break

        gene_idx, gene_name, total = item

        # Get gene length for display
        gene_len = _gene_length_from_csv(netseq_dir, gene_name)
        popsize = int(4 + 3 * math.log(gene_len)) if gene_len > 0 else 0
        n_runs = config_dict["n_runs"]
        n_threads = popsize * n_runs

        print(f"[{device_label}] Starting {gene_name} ({gene_idx+1}/{total}, "
              f"{gene_len} bp, popsize={popsize}, {n_threads} threads)", flush=True)

        t0 = time.time()
        result_path = gene_result_path(output_dir, gene_name)

        try:
            result = run_cmaes_for_gene(
                gene_name=gene_name,
                config=config,
                device_id=device_id,
                stream=stream,
            )
            save_result(result_path, result)

            mse = result["final_mse"]
            rms = result["final_rms"]
            gens = result["generations"]
            naive_mse = result["history"]["f_best"][0] if result["history"]["f_best"] else mse
            improvement = (naive_mse - mse) / naive_mse * 100 if naive_mse > 0 else 0.0
            elapsed = time.time() - t0

            print(f"[{device_label}] Finished {gene_name}: MSE={mse:.2f} "
                  f"(\u2193{improvement:.1f}% from naive), RMS={rms:.2f}, "
                  f"{gens} gens, {elapsed:.1f}s \u2192 saved", flush=True)

            # Send summary back to main process
            result_queue.put({
                "gene_name": gene_name,
                "gene_idx": gene_idx,
                "gene_length": result["gene_length"],
                "KRutLoading": result["KRutLoading"],
                "kRiboLoading": result["kRiboLoading"],
                "final_mse": mse,
                "final_rms": rms,
                "generations": gens,
                "early_stopped": result["early_stopped"],
                "wall_time": result["wall_time"],
                "elapsed": elapsed,
                "device": device_label,
                "status": "ok",
            })

        except Exception as e:
            elapsed = time.time() - t0
            error_info = {
                "gene_name": gene_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            save_result(gene_failed_path(output_dir, gene_name), error_info)
            print(f"[{device_label}] FAILED {gene_name}: {e}", flush=True)
            result_queue.put({
                "gene_name": gene_name,
                "gene_idx": gene_idx,
                "elapsed": elapsed,
                "device": device_label,
                "status": "failed",
                "error": str(e),
            })


def _check_cuda():
    """Check CUDA availability without polluting the main process CUDA state."""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "from numba import cuda; print(cuda.is_available()); print(len(cuda.gpus))"],
            capture_output=True, text=True, timeout=30,
        )
        lines = result.stdout.strip().split("\n")
        available = lines[0].strip() == "True"
        count = int(lines[1].strip()) if available else 0
        return available, count
    except Exception:
        return False, 0


def run_genome_wide(args):
    cuda_available, n_cuda_gpus = _check_cuda()

    script_dir = Path(__file__).resolve().parent
    netseq_dir = script_dir / "NETSEQ_gene"
    output_dir = Path(args.output)
    cmaes_dir = output_dir / "cmaes"
    cmaes_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "genome_wide_summary.csv"

    # Discover genes
    gene_filter = None
    if args.genes:
        gene_filter = [g.strip() for g in args.genes.split(",")]
    genes = discover_genes(netseq_dir, gene_filter)

    if not genes:
        print("No genes found.")
        return

    # Resume: skip genes with existing results in cmaes/ subdirectory
    if not args.force:
        already_done = set()
        for g in genes:
            if gene_result_path(cmaes_dir, g).exists():
                already_done.add(g)
        if already_done:
            print(f"Resuming: skipping {len(already_done)} already-completed genes")
            genes = [g for g in genes if g not in already_done]

    if not genes:
        print("All genes already processed. Use --force to re-run.")
        return

    # Sort genes by length (descending) to process longest first
    def _csv_line_count(path):
        try:
            with open(path) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    genes.sort(
        key=lambda g: _csv_line_count(netseq_dir / f"NETSEQ_{g}.csv"),
        reverse=True,
    )

    total = len(genes)

    # Determine parallelism
    n_gpus = min(args.gpu, n_cuda_gpus) if args.gpu > 0 else 0
    if args.gpu > 0 and not cuda_available:
        print("Warning: --gpu requested but CUDA not available. Falling back to CPU.")
        n_gpus = 0
    cpu_nt = args.cpu_nt
    n_cpu_workers = 1 if cpu_nt > 0 else 0
    # If no GPU and no CPU workers, create 1 CPU fallback worker
    if n_gpus == 0 and n_cpu_workers == 0:
        n_cpu_workers = 1
        if cpu_nt == 0:
            cpu_nt = os.cpu_count() or 1
    n_workers = n_gpus + n_cpu_workers

    n_runs = 200  # uniform schedule

    # Config dict (picklable, sent to worker processes)
    config_dict = {
        "sigma0": 0.1,
        "max_generations": args.max_gens,
        "use_gpu": True,
        "dt": 0.2,
        "n_runs": n_runs,
        "early_stop_window": 30,
        "early_stop_threshold": 0.005,
        "cpu_nt": None,
    }

    # Config summary
    print("=== Genome-wide CMA-ES Deconvolution ===")
    print(f"Output:       {cmaes_dir}")
    print(f"dt:           {config_dict['dt']}")
    print(f"n_runs:       {n_runs}")
    print(f"sigma0:       {config_dict['sigma0']}")
    print(f"max_gens:     {config_dict['max_generations']}")
    print(f"popsize:      auto (4 + 3*ln(geneLength))")
    print(f"early_stop:   {config_dict['early_stop_window']} gens, "
          f"{config_dict['early_stop_threshold']*100:.1f}% threshold")
    gpu_list = ", ".join(f"GPU-{i}" for i in range(n_gpus)) if n_gpus > 0 else "none"
    print(f"GPUs:         {n_gpus} ({gpu_list})" if n_gpus > 0 else "GPUs:         0")
    if cpu_nt > 0:
        print(f"CPU threads:  {cpu_nt}")
    else:
        print(f"CPU threads:  0")
    print(f"Total workers: {n_workers}")
    print()
    print(f"Processing {total} genes \u2192 {cmaes_dir}")
    print(flush=True)

    # Shared generation counter (atomically incremented by all workers)
    gen_counter = Value("i", 0)

    # Build work queue (multiprocessing-safe)
    gene_queue = MPQueue()
    result_queue = MPQueue()

    for i, gene in enumerate(genes):
        gene_queue.put((i, gene, total))

    # Poison pills
    for _ in range(n_workers):
        gene_queue.put(None)

    # Launch worker processes
    processes = []

    # GPU workers
    for w in range(n_gpus):
        gpu_config = dict(config_dict, use_gpu=True)
        p = Process(
            target=worker_process,
            args=(gene_queue, result_queue, cmaes_dir, gpu_config, w, netseq_dir, gen_counter),
        )
        p.start()
        processes.append(p)

    # CPU worker
    if n_cpu_workers > 0:
        cpu_config = dict(config_dict, use_gpu=False, cpu_nt=cpu_nt)
        p = Process(
            target=worker_process,
            args=(gene_queue, result_queue, cmaes_dir, cpu_config, -1, netseq_dir, gen_counter),
        )
        p.start()
        processes.append(p)

    # Register SIGTERM handler to clean up child processes on kill
    def _sigterm_handler(signum, frame):
        print("\nReceived SIGTERM — terminating worker processes...", flush=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Progress reporting from main process
    t_start = time.time()
    completed = 0
    rms_sum = 0.0
    total_gens_completed = 0  # from finished genes (for avg gens/gene)
    last_periodic_print = t_start

    def _print_eta(tag=""):
        total_elapsed = time.time() - t_start
        gens_so_far = gen_counter.value
        remaining_genes = total - completed
        mean_rms = rms_sum / completed if completed > 0 else 0.0

        if gens_so_far > 0 and total_elapsed > 0:
            gens_per_min = gens_so_far / total_elapsed * 60
            avg_gens_per_gene = total_gens_completed / completed if completed > 0 else args.max_gens
            remaining_gens = avg_gens_per_gene * remaining_genes
            eta_h = remaining_gens / (gens_so_far / total_elapsed) / 3600
            print(f"[{completed}/{total}] {gens_per_min:.0f} gens/min ({gens_so_far} gens) | "
                  f"ETA: {eta_h:.1f}h | mean_RMS={mean_rms:.4f}", flush=True)

    while any(p.is_alive() for p in processes) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1.0)
        except Exception:
            # No gene finished — check if we should print periodic ETA
            now = time.time()
            if now - last_periodic_print >= 60:
                last_periodic_print = now
                _print_eta()
            continue

        completed += 1

        if result["status"] == "ok":
            rms_sum += result["final_rms"]
            total_gens_completed += result["generations"]

            # Append to summary CSV
            append_summary_row(summary_csv, {
                "gene": result["gene_name"],
                "gene_length": result["gene_length"],
                "KRutLoading": result["KRutLoading"],
                "kRiboLoading": result["kRiboLoading"],
                "final_mse": f"{result['final_mse']:.6f}",
                "final_rms": f"{result['final_rms']:.4f}",
                "generations": result["generations"],
                "early_stopped": result["early_stopped"],
                "wall_time": f"{result['wall_time']:.1f}",
            })

        _print_eta()
        last_periodic_print = time.time()

    for p in processes:
        p.join()

    total_time = time.time() - t_start
    print(f"\nDone. {completed} genes in {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Results: {cmaes_dir}")
    print(f"Summary: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="Genome-wide CMA-ES deconvolution")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results (default: results/)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Number of GPUs to use (default: 0)")
    parser.add_argument("--cpu-nt", type=int, default=0,
                        help="Number of CPU threads for hybrid mode (default: 0, creates 1 CPU worker using N threads)")
    parser.add_argument("--max-gens", type=int, default=500,
                        help="Maximum CMA-ES generations per gene (default: 500)")
    parser.add_argument("--genes", type=str, default=None,
                        help="Comma-separated gene names to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing of already-completed genes")
    args = parser.parse_args()
    run_genome_wide(args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
