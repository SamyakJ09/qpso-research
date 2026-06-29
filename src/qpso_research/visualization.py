"""
Visualization functions for PSO vs QPSO comparison.

Provides:
  - plot_comparison_4panel(): signature 4-panel comparison plot
  - plot_convergence(): single-algorithm convergence curve
  - plot_multirun_boxplot(): boxplot for multi-run statistical analysis
"""

import matplotlib
matplotlib.use("Agg")  # Prevent Tk/GUI thread crash from Qiskit threads

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path


def plot_comparison_4panel(
    pso_result: dict,
    qpso_result: dict,
    fn_name: str,
    dimensions: int,
    goal_label: str,
    max_iter: int,
    success_threshold: float = 1e-6,
    pso_inertia_start: float = 0.9,
    pso_inertia_end: float = 0.4,
    qpso_mode: str = "math",
    save_path: str | Path | None = None,
) -> None:
    """
    Create the signature 4-panel comparison plot:
      Panel 1: Convergence comparison (best + avg)
      Panel 2: Iterations-to-success bar chart
      Panel 3: mbest distance over time
      Panel 4: Inertia / Beta decay curves
    """
    mode_label = {"math": "Math", "full": "Qiskit", "hybrid": "Hybrid"}
    qlabel = mode_label.get(qpso_mode, qpso_mode)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Classical PSO  vs  QPSO ({qlabel}) — {fn_name} ({dimensions}D)\n"
        f"Goal: {goal_label}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    iters = range(1, max_iter + 1)

    pso_success = pso_result["iter_to_success"]
    qpso_success = qpso_result["iter_to_success"]

    # ── Panel 1: Convergence ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(iters, pso_result["best_history"], color="tomato", linewidth=2,
             label="PSO  — Global Best")
    ax1.plot(iters, qpso_result["best_history"], color="royalblue", linewidth=2,
             label=f"QPSO — Global Best ({qlabel})")
    ax1.plot(iters, pso_result["avg_history"], color="tomato", linewidth=1,
             linestyle="--", alpha=0.45, label="PSO  — Swarm Avg")
    ax1.plot(iters, qpso_result["avg_history"], color="royalblue", linewidth=1,
             linestyle="--", alpha=0.45, label="QPSO — Swarm Avg")
    ax1.axhline(success_threshold, color="green", linestyle=":", linewidth=1.5,
                label=f"Goal threshold ({success_threshold:.0e})")
    if pso_success:
        ax1.axvline(pso_success, color="tomato", linestyle=":", alpha=0.8,
                    label=f"PSO  reaches goal @ iter {pso_success}")
    if qpso_success:
        ax1.axvline(qpso_success, color="royalblue", linestyle=":", alpha=0.8,
                    label=f"QPSO reaches goal @ iter {qpso_success}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness Score (log scale)")
    ax1.set_title("Convergence Comparison\n(lower = better  |  reach green line = success)")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, which="both", alpha=0.25)

    # ── Panel 2: Speedup bar chart ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    bar_vals = [
        pso_success if pso_success else max_iter,
        qpso_success if qpso_success else max_iter,
    ]
    bars = ax2.bar(["PSO", f"QPSO\n({qlabel})"], bar_vals,
                   color=["tomato", "royalblue"], width=0.5, edgecolor="white")
    for bar, val in zip(bars, bar_vals):
        lbl = str(val) if val < max_iter else "DNF"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max_iter * 0.02,
                 lbl, ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Iterations to reach goal")
    ax2.set_title("Iterations to Success\n(fewer = faster)")
    ax2.set_ylim(0, max_iter * 1.2)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: mbest distance ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(iters, pso_result["mbest_history"], color="tomato", linewidth=1.8,
             label="PSO  — mbest distance to goal")
    ax3.plot(iters, qpso_result["mbest_history"], color="royalblue", linewidth=1.8,
             label="QPSO — mbest distance to goal")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Distance from mbest to goal")
    ax3.set_title("Mean Best Position (mbest) Convergence\n"
                  "QPSO: mbest shapes quantum well width each iteration")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    # ── Panel 4: Inertia / Beta decay ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    pso_inertia = [pso_inertia_start - (i / max(max_iter - 1, 1)) *
                   (pso_inertia_start - pso_inertia_end) for i in range(max_iter)]
    ax4.plot(iters, pso_inertia, color="tomato", linewidth=1.8, label="PSO inertia w")
    ax4.plot(iters, qpso_result["beta_history"], color="royalblue", linewidth=1.8,
             label="QPSO Beta")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Parameter value")
    ax4.set_title("Exploration Control Decay\nPSO: inertia w  |  QPSO: Beta")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_multirun_boxplot(
    pso_scores: list[float],
    qpso_scores: list[float],
    benchmark_name: str,
    dims: int,
    qpso_mode: str = "math",
    save_path: str | Path | None = None,
) -> None:
    """
    Boxplot comparing final scores across multiple runs.
    Used for statistical analysis visualization.
    """
    mode_label = {"math": "Math", "full": "Qiskit", "hybrid": "Hybrid"}
    qlabel = mode_label.get(qpso_mode, qpso_mode)

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [pso_scores, qpso_scores]
    bp = ax.boxplot(data, labels=["PSO", f"QPSO ({qlabel})"],
                    patch_artist=True, widths=0.4)
    bp["boxes"][0].set_facecolor("tomato")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("royalblue")
    bp["boxes"][1].set_alpha(0.6)

    ax.set_ylabel("Final Best Score")
    ax.set_title(f"{benchmark_name} ({dims}D) — {len(pso_scores)} runs")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150)
    plt.close()
