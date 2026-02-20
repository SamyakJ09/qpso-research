"""
Full publication study: multi-benchmark, multi-dimension, multi-run comparison.

Runs PSO and QPSO across all configured benchmarks and dimensions,
performs Wilcoxon signed-rank tests, saves all results, and generates
comparison plots and statistical tables.

Usage:
  python experiments/run_full_study.py --config configs/default.yaml
  python experiments/run_full_study.py --config configs/publication.yaml
"""

import argparse
from pathlib import Path

import numpy as np

from qpso_research.benchmarks import get_benchmark
from qpso_research.config import load_config, ExperimentConfig
from qpso_research.pso import PSO
from qpso_research.qpso import QPSO
from qpso_research.statistics import wilcoxon_test, format_results_table
from qpso_research.visualization import plot_multirun_boxplot
from qpso_research.results import save_run_result, save_experiment_summary


def run_single(cfg, bench, dims, mode, seed):
    """Run a single PSO + QPSO trial with a given seed."""
    np.random.seed(seed)
    pso = PSO(
        objective_fn=bench.func,
        dimensions=dims,
        bounds=bench.bounds,
        num_particles=cfg.num_particles,
        max_iterations=cfg.max_iterations,
        inertia_start=cfg.pso.inertia_start,
        inertia_end=cfg.pso.inertia_end,
        cognitive_coef=cfg.pso.cognitive_coef,
        social_coef=cfg.pso.social_coef,
        velocity_clamp=cfg.pso.velocity_clamp,
        stagnation_limit=cfg.stagnation_limit,
        success_threshold=cfg.success_threshold,
    )
    goal_coords = bench.optimal_position(dims).tolist()
    pso_result = pso.optimize(verbose=False, goal_coords=goal_coords)

    np.random.seed(seed)
    qpso = QPSO(
        objective_fn=bench.func,
        dimensions=dims,
        bounds=bench.bounds,
        num_particles=cfg.num_particles,
        max_iterations=cfg.max_iterations,
        beta_start=cfg.qpso.beta_start,
        beta_end=cfg.qpso.beta_end,
        cognitive_coef=cfg.qpso.cognitive_coef,
        social_coef=cfg.qpso.social_coef,
        stagnation_limit=cfg.stagnation_limit,
        success_threshold=cfg.success_threshold,
        mode=mode,
        shots=cfg.qpso.shots,
    )
    qpso_result = qpso.optimize(verbose=False, goal_coords=goal_coords)

    return pso_result, qpso_result


def main():
    parser = argparse.ArgumentParser(description="Full PSO vs QPSO publication study")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg.results_dir)

    all_results = {}  # {benchmark: {algorithm: {"mean", "std", "best", "scores"}}}
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
                    pso_result, qpso_result = run_single(cfg, bench, dims, mode, seed)

                    pso_scores.append(pso_result["best_score"])
                    qpso_scores.append(qpso_result["best_score"])

                    # Save individual run CSVs
                    save_run_result(
                        pso_result,
                        results_dir / "raw" / f"pso_{bench_name}_{dims}d_run{run}.csv",
                    )
                    save_run_result(
                        qpso_result,
                        results_dir / "raw" / f"qpso_{mode}_{bench_name}_{dims}d_run{run}.csv",
                    )

                # ── Statistics ─────────────────────────────────────
                pso_arr = np.array(pso_scores)
                qpso_arr = np.array(qpso_scores)

                wilcoxon = wilcoxon_test(pso_scores, qpso_scores)

                print(f"    PSO  — mean: {pso_arr.mean():.2e}  std: {pso_arr.std():.2e}")
                print(f"    QPSO — mean: {qpso_arr.mean():.2e}  std: {qpso_arr.std():.2e}")
                print(f"    Wilcoxon p={wilcoxon.p_value:.4f} "
                      f"{'*** SIGNIFICANT' if wilcoxon.significant else '(not significant)'}")

                # Store for summary
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

                # ── Boxplot ────────────────────────────────────────
                plot_multirun_boxplot(
                    pso_scores=pso_scores,
                    qpso_scores=qpso_scores,
                    benchmark_name=bench.name,
                    dims=dims,
                    qpso_mode=mode,
                    save_path=results_dir / "figures" / f"boxplot_{mode}_{bench_name}_{dims}d.png",
                )

    # ── Save summary ──────────────────────────────────────────────
    save_experiment_summary(summary_data, results_dir / "summary" / "experiment_summary.json")

    # ── Print results table ───────────────────────────────────────
    bench_keys = list(all_results.keys())
    if bench_keys:
        print("\n" + "=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)
        table = format_results_table(all_results, bench_keys)
        print(table)

        # Save table
        table_path = results_dir / "summary" / "results_table.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            f.write(table)
        print(f"\n  Table saved -> {table_path}")

    print(f"\n  All results saved to {results_dir}/")


if __name__ == "__main__":
    main()
