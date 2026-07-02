"""
Console script entry points for qpso-compare and qpso-study.

Registered in pyproject.toml [project.scripts] so users can run
experiments without invoking ``python`` directly — which avoids the
Windows Microsoft Store app-execution-alias issue entirely.

Also used by ``__main__.py`` for ``python -m qpso_research`` execution.
"""

import argparse
from pathlib import Path

import numpy as np

from .benchmarks import get_benchmark
from .config import load_config, ExperimentConfig
from .pso import PSO
from .qpso import QPSO
from .statistics import wilcoxon_test, format_results_table
from .visualization import plot_comparison_4panel, plot_multirun_boxplot
from .results import save_run_result, save_experiment_summary


# ── Shared builders ──────────────────────────────────────────────────

def _build_pso(cfg: ExperimentConfig, func, dims: int, bounds):
    """Construct a PSO instance from experiment config."""
    return PSO(
        objective_fn=func, dimensions=dims, bounds=bounds,
        num_particles=cfg.num_particles, max_iterations=cfg.max_iterations,
        inertia_start=cfg.pso.inertia_start, inertia_end=cfg.pso.inertia_end,
        cognitive_coef=cfg.pso.cognitive_coef, social_coef=cfg.pso.social_coef,
        velocity_clamp=cfg.pso.velocity_clamp,
        stagnation_limit=cfg.stagnation_limit,
        success_threshold=cfg.success_threshold,
    )


def _build_qpso(cfg: ExperimentConfig, func, dims: int, bounds, mode: str):
    """Construct a QPSO instance from experiment config."""
    return QPSO(
        objective_fn=func, dimensions=dims, bounds=bounds,
        num_particles=cfg.num_particles, max_iterations=cfg.max_iterations,
        beta_start=cfg.qpso.beta_start, beta_end=cfg.qpso.beta_end,
        stagnation_limit=cfg.stagnation_limit,
        success_threshold=cfg.success_threshold,
        mode=mode, shots=cfg.qpso.shots,
        momentum_decay=cfg.qpso.momentum_decay,
        momentum_weight=cfg.qpso.momentum_weight,
    )


# ── Single-benchmark comparison ──────────────────────────────────────

def _run_one_comparison(cfg: ExperimentConfig, benchmark_name: str,
                        dims: int, mode: str):
    """Run PSO vs QPSO on a single benchmark and produce comparison output."""
    bench = get_benchmark(benchmark_name)
    goal_coords = bench.optimal_position(dims).tolist()

    print("\n" + "=" * 70)
    print(f"  {bench.name}  ({dims}D)  |  Goal: {goal_coords}")
    print(f"  Mode: {mode}  |  Particles: {cfg.num_particles}  |  Iters: {cfg.max_iterations}")
    print("=" * 70)

    print("\n--- Classical PSO ---")
    np.random.seed(cfg.seed)
    pso_result = _build_pso(cfg, bench.func, dims, bench.bounds).optimize(
        verbose=cfg.verbose, goal_coords=goal_coords,
    )

    print(f"\n--- QPSO (mode={mode}) ---")
    np.random.seed(cfg.seed)
    qpso_result = _build_qpso(cfg, bench.func, dims, bench.bounds, mode).optimize(
        verbose=cfg.verbose, goal_coords=goal_coords,
    )

    # Summary
    print(f"\n{'-' * 70}")
    print(f"  {'Metric':<35} {'PSO':>12}  {'QPSO':>12}")
    print(f"{'-' * 70}")
    print(f"  {'Final best score':<35} {pso_result['best_score']:>12.2e}  "
          f"{qpso_result['best_score']:>12.2e}")
    pso_its = pso_result["iter_to_success"]
    qpso_its = qpso_result["iter_to_success"]
    print(f"  {'Iters to reach goal':<35} "
          f"{str(pso_its) if pso_its else 'NOT REACHED':>12}  "
          f"{str(qpso_its) if qpso_its else 'NOT REACHED':>12}")
    print(f"  {'Time elapsed (s)':<35} {pso_result['elapsed_time']:>12.3f}  "
          f"{qpso_result['elapsed_time']:>12.3f}")
    if pso_its and qpso_its:
        print(f"\n  >>> QPSO reached goal {pso_its / qpso_its:.2f}x FASTER than PSO <<<")
    print(f"{'-' * 70}")

    # Save
    results_dir = Path(cfg.results_dir)
    save_run_result(pso_result, results_dir / "raw" / f"pso_{benchmark_name}_{dims}d.csv")
    save_run_result(qpso_result, results_dir / "raw" / f"qpso_{mode}_{benchmark_name}_{dims}d.csv")

    plot_path = results_dir / "figures" / f"pso_vs_qpso_{mode}_{benchmark_name}_{dims}d.png"
    plot_comparison_4panel(
        pso_result=pso_result, qpso_result=qpso_result,
        fn_name=bench.name, dimensions=dims, goal_label=str(goal_coords),
        max_iter=cfg.max_iterations, success_threshold=cfg.success_threshold,
        pso_inertia_start=cfg.pso.inertia_start,
        pso_inertia_end=cfg.pso.inertia_end,
        qpso_mode=mode, save_path=plot_path,
    )
    print(f"  Plot saved -> {plot_path}")


def run_comparison():
    """Entry point for ``qpso-compare`` command."""
    parser = argparse.ArgumentParser(description="PSO vs QPSO single-benchmark comparison")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--benchmark", type=str, default="sphere", help="Benchmark function name")
    parser.add_argument("--dims", type=int, default=2, help="Number of dimensions")
    parser.add_argument("--mode", type=str, default="math",
                        choices=["math", "full", "hybrid"], help="QPSO randomness mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else ExperimentConfig()
    if args.seed is not None:
        cfg.seed = args.seed

    _run_one_comparison(cfg, args.benchmark, args.dims, args.mode)


# ── Full multi-benchmark study ───────────────────────────────────────

def _run_single_trial(cfg, bench, dims, mode, seed):
    """Run a single PSO + QPSO trial with a given seed."""
    goal_coords = bench.optimal_position(dims).tolist()

    np.random.seed(seed)
    pso_result = _build_pso(cfg, bench.func, dims, bench.bounds).optimize(
        verbose=False, goal_coords=goal_coords,
    )

    np.random.seed(seed)
    qpso_result = _build_qpso(cfg, bench.func, dims, bench.bounds, mode).optimize(
        verbose=False, goal_coords=goal_coords,
    )

    return pso_result, qpso_result


def run_full_study():
    """Entry point for ``qpso-study`` command."""
    parser = argparse.ArgumentParser(description="Full PSO vs QPSO publication study")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg.results_dir)

    all_results = {}
    summary_data = {}

    for mode in cfg.qpso_modes:
        print(f"\n{'#' * 70}")
        print(f"  QPSO Mode: {mode}")
        print(f"{'#' * 70}")

        for bench_name in cfg.benchmarks:
            bench = get_benchmark(bench_name)
            for dims in cfg.dimensions:
                key = f"{bench_name}_{dims}d"
                pso_scores = []
                qpso_scores = []

                print(f"\n  {bench.name} ({dims}D) — {cfg.num_runs} runs...")

                for run in range(cfg.num_runs):
                    seed = cfg.seed + run
                    pso_result, qpso_result = _run_single_trial(
                        cfg, bench, dims, mode, seed,
                    )
                    pso_scores.append(pso_result["best_score"])
                    qpso_scores.append(qpso_result["best_score"])

                    save_run_result(
                        pso_result,
                        results_dir / "raw" / f"pso_{bench_name}_{dims}d_run{run}.csv",
                    )
                    save_run_result(
                        qpso_result,
                        results_dir / "raw" / f"qpso_{mode}_{bench_name}_{dims}d_run{run}.csv",
                    )

                pso_arr = np.array(pso_scores)
                qpso_arr = np.array(qpso_scores)
                wilcoxon = wilcoxon_test(pso_scores, qpso_scores)

                print(f"    PSO  — mean: {pso_arr.mean():.2e}  std: {pso_arr.std():.2e}")
                print(f"    QPSO — mean: {qpso_arr.mean():.2e}  std: {qpso_arr.std():.2e}")
                print(f"    Wilcoxon p={wilcoxon.p_value:.4f} "
                      f"{'*** SIGNIFICANT' if wilcoxon.significant else '(not significant)'}")

                if key not in all_results:
                    all_results[key] = {}
                all_results[key]["PSO"] = {
                    "mean": float(pso_arr.mean()),
                    "std": float(pso_arr.std()),
                    "best": float(pso_arr.min()),
                    "scores": pso_scores,
                }
                all_results[key][f"QPSO ({mode})"] = {
                    "mean": float(qpso_arr.mean()),
                    "std": float(qpso_arr.std()),
                    "best": float(qpso_arr.min()),
                    "scores": qpso_scores,
                }

                summary_data[key] = {
                    "benchmark": bench_name,
                    "dimensions": dims,
                    "mode": mode,
                    "num_runs": cfg.num_runs,
                    "pso_mean": float(pso_arr.mean()),
                    "pso_std": float(pso_arr.std()),
                    "qpso_mean": float(qpso_arr.mean()),
                    "qpso_std": float(qpso_arr.std()),
                    "wilcoxon_p": wilcoxon.p_value,
                    "wilcoxon_significant": wilcoxon.significant,
                }

                plot_multirun_boxplot(
                    pso_scores=pso_scores, qpso_scores=qpso_scores,
                    benchmark_name=bench.name, dims=dims, qpso_mode=mode,
                    save_path=results_dir / "figures" / f"boxplot_{mode}_{bench_name}_{dims}d.png",
                )

    save_experiment_summary(summary_data, results_dir / "summary" / "experiment_summary.json")

    bench_keys = list(all_results.keys())
    if bench_keys:
        print("\n" + "=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)
        table = format_results_table(all_results, bench_keys)
        print(table)

        table_path = results_dir / "summary" / "results_table.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table)
        print(f"\n  Table saved -> {table_path}")

    print(f"\n  All results saved to {results_dir}/")
