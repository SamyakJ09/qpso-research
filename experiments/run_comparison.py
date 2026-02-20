"""
Single-benchmark comparison: PSO vs QPSO.

Replaces the if __name__ == "__main__" blocks from the original
pso_vs_qpsoTEST1/2/3 files. Runs PSO and QPSO on one benchmark,
produces a 4-panel comparison plot and prints a results summary.

Usage:
  python experiments/run_comparison.py --config configs/default.yaml --benchmark sphere --dims 2 --mode math
  python experiments/run_comparison.py --benchmark rastrigin --dims 5 --mode hybrid
"""

import argparse
from pathlib import Path

import numpy as np

from qpso_research.benchmarks import get_benchmark
from qpso_research.config import load_config, ExperimentConfig
from qpso_research.pso import PSO
from qpso_research.qpso import QPSO
from qpso_research.visualization import plot_comparison_4panel
from qpso_research.results import save_run_result


def run_comparison(cfg: ExperimentConfig, benchmark_name: str, dims: int, mode: str):
    """Run PSO vs QPSO on a single benchmark and produce comparison output."""
    bench = get_benchmark(benchmark_name)
    goal_coords = bench.optimal_position(dims).tolist()

    print("\n" + "=" * 70)
    print(f"  {bench.name}  ({dims}D)  |  Goal: {goal_coords}")
    print(f"  Mode: {mode}  |  Particles: {cfg.num_particles}  |  Iters: {cfg.max_iterations}")
    print("=" * 70)

    # ── PSO ────────────────────────────────────────────────────────
    print("\n--- Classical PSO ---")
    np.random.seed(cfg.seed)
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
    pso_result = pso.optimize(verbose=cfg.verbose, goal_coords=goal_coords)

    # ── QPSO ───────────────────────────────────────────────────────
    print(f"\n--- QPSO (mode={mode}) ---")
    np.random.seed(cfg.seed)
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
    qpso_result = qpso.optimize(verbose=cfg.verbose, goal_coords=goal_coords)

    # ── Results Summary ────────────────────────────────────────────
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
        speedup = pso_its / qpso_its
        print(f"\n  >>> QPSO reached goal {speedup:.2f}x FASTER than PSO <<<")
    print(f"{'-' * 70}")

    # ── Save results ───────────────────────────────────────────────
    results_dir = Path(cfg.results_dir)
    save_run_result(pso_result, results_dir / "raw" / f"pso_{benchmark_name}_{dims}d.csv")
    save_run_result(qpso_result, results_dir / "raw" / f"qpso_{mode}_{benchmark_name}_{dims}d.csv")

    # ── Plot ───────────────────────────────────────────────────────
    plot_path = results_dir / "figures" / f"pso_vs_qpso_{mode}_{benchmark_name}_{dims}d.png"
    plot_comparison_4panel(
        pso_result=pso_result,
        qpso_result=qpso_result,
        fn_name=bench.name,
        dimensions=dims,
        goal_label=str(goal_coords),
        max_iter=cfg.max_iterations,
        success_threshold=cfg.success_threshold,
        pso_inertia_start=cfg.pso.inertia_start,
        pso_inertia_end=cfg.pso.inertia_end,
        qpso_mode=mode,
        save_path=plot_path,
    )
    print(f"  Plot saved -> {plot_path}")

    return pso_result, qpso_result


def main():
    parser = argparse.ArgumentParser(description="PSO vs QPSO single-benchmark comparison")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--benchmark", type=str, default="sphere", help="Benchmark function name")
    parser.add_argument("--dims", type=int, default=2, help="Number of dimensions")
    parser.add_argument("--mode", type=str, default="math",
                        choices=["math", "full", "hybrid"], help="QPSO randomness mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = ExperimentConfig()

    if args.seed is not None:
        cfg.seed = args.seed

    run_comparison(cfg, args.benchmark, args.dims, args.mode)


if __name__ == "__main__":
    main()
