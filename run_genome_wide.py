#!/usr/bin/env python
"""
run_genome_wide.py
==================
Genome-wide CMA-ES deconvolution runner.

Iterates over all E. coli genes in NETSEQ_gene/, runs CMA-ES deconvolution
for each gene, and collects results into per-gene pickle files + a summary CSV.

Usage:
    python run_genome_wide.py --output results/ --device gpu
    python run_genome_wide.py --output results/ --device cpu --genes insQ,talB,aceA
    python run_genome_wide.py --output results/ --device gpu --force --streams-per-gpu 2
"""

import argparse
import csv
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from queue import Queue
from threading import Thread

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


def process_gene(gene_name: str, output_dir: Path, config, device_id: int, stream):
    """Run CMA-ES for one gene and save results. Returns (gene_name, result_or_error)."""
    from cmaes_multifidelity import run_cmaes_for_gene

    result_path = gene_result_path(output_dir, gene_name)
    try:
        result = run_cmaes_for_gene(
            gene_name=gene_name,
            config=config,
            device_id=device_id,
            stream=stream,
        )
        save_result(result_path, result)
        return gene_name, result
    except Exception as e:
        error_info = {
            "gene_name": gene_name,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        save_result(gene_failed_path(output_dir, gene_name), error_info)
        return gene_name, error_info


def worker_thread(gene_queue: Queue, results_list: list, output_dir: Path,
                  config, device_id: int, stream_idx: int):
    """Worker thread: pull genes from queue, process them."""
    from netseq_tasep_gpu import cuda_is_available

    # Create a CUDA stream if GPU is available
    stream = None
    if config.use_gpu and cuda_is_available():
        from numba import cuda
        cuda.select_device(device_id)
        stream = cuda.stream()

    while True:
        item = gene_queue.get()
        if item is None:  # poison pill
            gene_queue.task_done()
            break
        gene_idx, gene_name, total = item
        t0 = time.time()
        gene_name, result = process_gene(gene_name, output_dir, config, device_id, stream)
        elapsed = time.time() - t0
        results_list.append((gene_idx, gene_name, result, elapsed))
        gene_queue.task_done()


def run_genome_wide(args):
    from cmaes_multifidelity import CMAESConfig, NRunsSchedule, EarlyStopping
    from netseq_tasep_gpu import cuda_is_available, gpu_count

    script_dir = Path(__file__).resolve().parent
    netseq_dir = script_dir / "NETSEQ_gene"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "genome_wide_summary.csv"

    # Discover genes
    gene_filter = None
    if args.genes:
        gene_filter = [g.strip() for g in args.genes.split(",")]
    genes = discover_genes(netseq_dir, gene_filter)

    if not genes:
        print("No genes found.")
        return

    # Resume: skip genes with existing results
    if not args.force:
        already_done = set()
        for g in genes:
            if gene_result_path(output_dir, g).exists():
                already_done.add(g)
        if already_done:
            print(f"Resuming: skipping {len(already_done)} already-completed genes")
            genes = [g for g in genes if g not in already_done]

    if not genes:
        print("All genes already processed. Use --force to re-run.")
        return

    # Sort genes by length (descending) to process longest first,
    # avoiding stragglers at the end when most workers are idle.
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
    print(f"Processing {total} genes → {output_dir}")

    # Config
    use_gpu = args.device == "gpu" and cuda_is_available()
    if args.device == "gpu" and not cuda_is_available():
        print("Warning: GPU requested but CUDA not available. Falling back to CPU.")

    config = CMAESConfig(
        use_gpu=use_gpu,
        n_runs_schedule=NRunsSchedule(),
        early_stopping=EarlyStopping(),
    )

    # Determine parallelism
    n_gpus = min(args.n_gpus, gpu_count()) if use_gpu else 0
    streams_per_gpu = args.streams_per_gpu if use_gpu else 1
    n_workers = max(1, n_gpus * streams_per_gpu) if use_gpu else 1

    print(f"Device: {'GPU' if use_gpu else 'CPU'}, GPUs: {n_gpus}, "
          f"Streams/GPU: {streams_per_gpu}, Workers: {n_workers}")

    # Build work queue
    gene_queue = Queue()
    for i, gene in enumerate(genes):
        gene_queue.put((i, gene, total))

    # Poison pills
    for _ in range(n_workers):
        gene_queue.put(None)

    # Launch workers
    results_list = []
    threads = []
    for w in range(n_workers):
        if use_gpu:
            dev_id = w // streams_per_gpu
            s_idx = w % streams_per_gpu
        else:
            dev_id = 0
            s_idx = 0
        t = Thread(target=worker_thread,
                   args=(gene_queue, results_list, output_dir, config, dev_id, s_idx))
        t.start()
        threads.append(t)

    # Progress reporting
    t_start = time.time()
    completed = 0
    rms_sum = 0.0

    while any(t.is_alive() for t in threads):
        time.sleep(1.0)
        # Report newly completed
        while completed < len(results_list):
            gene_idx, gene_name, result, elapsed = results_list[completed]
            completed += 1

            if isinstance(result, dict) and "final_mse" in result:
                rms = result["final_rms"]
                mse = result["final_mse"]
                gens = result["generations"]
                early = result["early_stopped"]
                rms_sum += rms

                # Append to summary CSV
                append_summary_row(summary_csv, {
                    "gene": gene_name,
                    "gene_length": result["gene_length"],
                    "KRutLoading": result["KRutLoading"],
                    "kRiboLoading": result["kRiboLoading"],
                    "final_mse": f"{mse:.6f}",
                    "final_rms": f"{rms:.4f}",
                    "generations": gens,
                    "early_stopped": early,
                    "wall_time": f"{result['wall_time']:.1f}",
                })

                total_elapsed = time.time() - t_start
                avg_time = total_elapsed / completed
                remaining = total - completed
                eta_s = avg_time * remaining
                eta_h = eta_s / 3600
                mean_rms = rms_sum / completed

                print(f"[{completed}/{total}] {gene_name}: MSE={mse:.4f}, "
                      f"RMS={rms:.4f}, {elapsed:.1f}s | "
                      f"ETA: {eta_h:.1f}h | mean_RMS={mean_rms:.4f}")
            else:
                # Failed gene
                err_msg = result.get("error", "unknown") if isinstance(result, dict) else str(result)
                print(f"[{completed}/{total}] {gene_name}: FAILED - {err_msg}")

    for t in threads:
        t.join()

    total_time = time.time() - t_start
    print(f"\nDone. {completed} genes in {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Results: {output_dir}")
    print(f"Summary: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="Genome-wide CMA-ES deconvolution")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results (default: results/)")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="Computation device (default: gpu)")
    parser.add_argument("--genes", type=str, default=None,
                        help="Comma-separated gene names to process (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing of already-completed genes")
    parser.add_argument("--n-gpus", type=int, default=99,
                        help="Max GPUs to use (default: all available)")
    parser.add_argument("--streams-per-gpu", type=int, default=1,
                        help="CUDA streams per GPU (default: 1, concurrent streams don't improve throughput)")
    args = parser.parse_args()
    run_genome_wide(args)


if __name__ == "__main__":
    main()
