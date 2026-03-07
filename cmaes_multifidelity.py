"""
cmaes_multifidelity.py
======================
Multi-fidelity CMA-ES optimizer for TASEP deconvolution.

Features:
  - Direct initialization from experimental signal (D_init = S_exp)
  - GPU-aware N_RUNS schedule (200/400/600 based on sigma)
  - Adaptive early stopping on convergence plateau
  - GPU or CPU dispatch via netseq_tasep_gpu.objective_batch_gpu
"""

import math
import time
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cma
except ImportError:
    cma = None


# ---------------------------------------------------------------------------
# GPU-aware N_RUNS scheduler
# ---------------------------------------------------------------------------

@dataclass
class NRunsSchedule:
    """Multi-fidelity N_RUNS schedule based on sigma."""
    n_runs_low: int = 200
    n_runs_med: int = 400
    n_runs_high: int = 800
    sigma_thresh_med: float = 0.8   # fraction of sigma0
    sigma_thresh_high: float = 0.6  # fraction of sigma0
    max_threads_per_gene: int = 60000

    def get_n_runs(self, sigma, sigma0, popsize):
        """Return N_RUNS for current sigma."""
        ratio = sigma / sigma0 if sigma0 > 0 else 0
        if ratio > self.sigma_thresh_med:
            n = self.n_runs_low
        elif ratio > self.sigma_thresh_high:
            n = self.n_runs_med
        else:
            n = self.n_runs_high
        # Cap so popsize * n_runs <= max_threads_per_gene
        max_n = max(1, self.max_threads_per_gene // max(1, popsize))
        return min(n, max_n)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

@dataclass
class EarlyStopping:
    """Stop if relative improvement < threshold over a window of generations."""
    window: int = 30
    rel_threshold: float = 0.005  # 0.5%

    def should_stop(self, f_best_history):
        if len(f_best_history) < self.window:
            return False
        old_best = f_best_history[-self.window]
        new_best = f_best_history[-1]
        if old_best <= 0:
            return False
        rel_improvement = (old_best - new_best) / abs(old_best)
        return rel_improvement < self.rel_threshold


# ---------------------------------------------------------------------------
# CMA-ES configuration
# ---------------------------------------------------------------------------

@dataclass
class CMAESConfig:
    sigma0: float = 0.1
    max_generations: int = 500
    popsize: int | None = None
    cma_seed: int = 42
    use_warm_start: bool = False
    n_runs_schedule: NRunsSchedule = field(default_factory=NRunsSchedule)
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
    checkpoint_every: int = 25
    checkpoint_dir: str | None = None
    display_every: int = 1
    use_gpu: bool = True
    device_id: int = 0
    stream: object = None


# ---------------------------------------------------------------------------
# Main CMA-ES runner
# ---------------------------------------------------------------------------

def run_cmaes_for_gene(
    gene_name: str,
    config: CMAESConfig | None = None,
    device_id: int = 0,
    stream=None,
) -> dict:
    """
    Run CMA-ES deconvolution for a single gene.

    Parameters
    ----------
    gene_name : str
    config : CMAESConfig, optional (uses defaults if None)
    device_id : int, GPU device id
    stream : CUDA stream or None

    Returns
    -------
    dict with keys: D_best, theta_best, S_exp_norm, gene_name, gene_length,
                    history, final_mse, final_rms, generations, early_stopped,
                    wall_time, config
    """
    if cma is None:
        raise ImportError("pycma is required: pip install cma")

    from netseq_tasep_fast import _load_gene_parameters
    from netseq_tasep_gpu import objective_batch_gpu, cuda_is_available

    if config is None:
        config = CMAESConfig()

    # Override device/stream from arguments
    config.device_id = device_id
    if stream is not None:
        config.stream = stream

    # Load gene data
    base_params = _load_gene_parameters(gene_name)
    S_exp_norm = base_params["RNAP_dwellTimeProfile"].copy()
    gene_length = len(S_exp_norm)
    KRutLoading = base_params["KRutLoading"]

    # Initial guess: direct from experimental signal
    theta_init = np.log(np.maximum(S_exp_norm, 1e-10))

    # CMA-ES options
    opts = {
        "CMA_diagonal": True,
        "seed": config.cma_seed,
        "maxiter": config.max_generations,
        "verb_disp": 0,
        "verb_log": 0,
        "tolfun": 1e-8,
        "tolx": 1e-8,
    }
    if config.popsize is not None:
        opts["popsize"] = config.popsize

    es = cma.CMAEvolutionStrategy(theta_init, config.sigma0, opts)
    sigma0 = es.sigma

    # Checkpoint dir
    ckpt_dir = None
    if config.checkpoint_dir:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # History tracking
    history = {
        "gen": [],
        "f_best": [],
        "f_median": [],
        "sigma": [],
        "n_runs": [],
        "elapsed": [],
    }

    batch_seed = config.cma_seed * 100000
    gen_count = 0
    early_stopped = False
    t_start = time.time()

    use_gpu = config.use_gpu and cuda_is_available()

    while not es.stop():
        # Determine N_RUNS for this generation
        n_runs = config.n_runs_schedule.get_n_runs(es.sigma, sigma0, es.popsize)

        # Sample candidates
        candidates = es.ask()

        # Evaluate
        fitnesses = objective_batch_gpu(
            thetas=candidates,
            base_params=base_params,
            n_runs=n_runs,
            S_exp_norm=S_exp_norm,
            base_seed=batch_seed,
            device_id=config.device_id,
            stream=config.stream,
        )
        batch_seed += len(candidates) * n_runs

        es.tell(candidates, fitnesses)
        gen_count += 1
        elapsed = time.time() - t_start

        history["gen"].append(gen_count)
        history["f_best"].append(es.result.fbest)
        history["f_median"].append(float(np.median(fitnesses)))
        history["sigma"].append(es.sigma)
        history["n_runs"].append(n_runs)
        history["elapsed"].append(elapsed)

        # Progress display
        if config.display_every > 0 and gen_count % config.display_every == 0:
            eta = elapsed / gen_count * (config.max_generations - gen_count)
            print(f"  gen {gen_count:4d} | f_best={es.result.fbest:.6f} | "
                  f"f_med={float(np.median(fitnesses)):.6f} | "
                  f"sigma={es.sigma:.4f} | n_runs={n_runs} | "
                  f"{elapsed:.0f}s | ETA {eta:.0f}s")

        # Early stopping check
        if config.early_stopping.should_stop(history["f_best"]):
            early_stopped = True
            print(f"  Early stopping at generation {gen_count} "
                  f"(< {config.early_stopping.rel_threshold*100:.1f}% improvement "
                  f"over {config.early_stopping.window} gens)")
            break

        # Checkpoint
        if ckpt_dir and gen_count % config.checkpoint_every == 0:
            _save_checkpoint(ckpt_dir / f"gen_{gen_count:04d}.pkl", es, history, batch_seed)
            if config.display_every > 0:
                print(f"  >> Checkpoint saved: gen_{gen_count:04d}.pkl")

    wall_time = time.time() - t_start

    # Extract best result
    theta_best = es.result.xbest
    D_best = np.exp(theta_best)
    D_best_norm = D_best / np.mean(D_best)

    result = {
        "D_best": D_best_norm,
        "theta_best": theta_best,
        "S_exp_norm": S_exp_norm,
        "gene_name": gene_name,
        "gene_length": gene_length,
        "KRutLoading": KRutLoading,
        "kRiboLoading": base_params["kRiboLoading"],
        "history": history,
        "final_mse": float(es.result.fbest),
        "final_rms": float(np.sqrt(es.result.fbest)),
        "generations": gen_count,
        "early_stopped": early_stopped,
        "wall_time": wall_time,
    }

    # Save final checkpoint
    if ckpt_dir:
        _save_checkpoint(ckpt_dir / "final.pkl", es, history, batch_seed, extra=result)

    return result


def _save_checkpoint(path, es, history, batch_seed, extra=None):
    data = {"es": es, "history": history, "batch_seed": batch_seed}
    if extra:
        data.update(extra)
    with open(path, "wb") as f:
        pickle.dump(data, f)
