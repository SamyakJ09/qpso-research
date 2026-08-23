"""
Generate the publication figures referenced in paper/sections/results.tex.

Reads the per-run convergence CSVs written by ``qpso-study`` (results/raw/)
and produces:

    paper/figures/convergence_grid.png       -> fig:convergence-curves
    paper/figures/convergence_rosenbrock.png -> fig:convergence-rosenbrock

The Schwefel box plot (fig:boxplots-schwefel) is produced directly by the
study run as results/figures/boxplot_math_schwefel_20d.png.

Each convergence panel plots the median global-best trajectory across all
available runs with a shaded inter-quartile range (IQR), matching the
PSO=tomato / QPSO=royalblue palette used elsewhere in the package.

Usage:
  python experiments/make_paper_figures.py            # defaults (grid at n=10)
  python experiments/make_paper_figures.py --dim 10
    python experiments/make_paper_figures.py --rosenbrock-dim 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Serif + gridded style so the figures read as part of the LaTeX serif body
# (matches the publication document). Vector PDF is the primary output.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.frameon": False,
    "pdf.fonttype": 42,   # embed TrueType (editable/searchable text in the PDF)
})

RAW = Path("results/raw")
FIG = Path("paper/figures")   # tracked, so the paper builds from a clean clone
FLOOR = 1e-10  # clamp for log-scale display (scores can reach exact 0)


def _savefig_all(fig, stem: str) -> None:
    """Save a figure as vector PDF (for LaTeX) plus SVG and PNG copies."""
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)

PSO_COLOR = "tomato"
QPSO_COLOR = "royalblue"

BENCHMARK_TITLES = {
    "sphere": "Sphere",
    "rastrigin": "Rastrigin",
    "rosenbrock": "Rosenbrock",
    "ackley": "Ackley",
    "griewank": "Griewank",
    "schwefel": "Schwefel",
}


def _load_histories(prefix: str, bench: str, dim: int) -> np.ndarray | None:
    """Stack the best_score column from every run CSV -> (runs, iters)."""
    rows = []
    for run in range(200):  # generous upper bound; stops at first gap
        fn = RAW / f"{prefix}_{bench}_{dim}d_run{run}.csv"
        if not fn.exists():
            break
        with open(fn) as f:
            rows.append([float(r["best_score"]) for r in csv.DictReader(f)])
    if not rows:
        return None
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows])


def _plot_panel(ax, bench: str, dim: int) -> bool:
    pso = _load_histories("pso", bench, dim)
    qpso = _load_histories("qpso_math", bench, dim)
    if pso is None or qpso is None:
        ax.set_visible(False)
        return False

    x = np.arange(1, pso.shape[1] + 1)
    for data, color, label in ((pso, PSO_COLOR, "PSO"),
                               (qpso, QPSO_COLOR, "QPSO")):
        med = np.clip(np.median(data, axis=0), FLOOR, None)
        q25 = np.clip(np.percentile(data, 25, axis=0), FLOOR, None)
        q75 = np.clip(np.percentile(data, 75, axis=0), FLOOR, None)
        ax.plot(x, med, color=color, lw=1.8, label=label)
        ax.fill_between(x, q25, q75, color=color, alpha=0.18)

    ax.set_yscale("log")
    ax.set_title(f"{BENCHMARK_TITLES.get(bench, bench)} ($n={dim}$)", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    n_runs = pso.shape[0]
    return n_runs


def make_grid(dim: int) -> None:
    benches = ["sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "schwefel"]
    missing = [bench for bench in benches
               if _load_histories("pso", bench, dim) is None
               or _load_histories("qpso_math", bench, dim) is None]
    if missing:
        raise RuntimeError(
            f"Missing convergence data for {dim}D: {', '.join(missing)}. "
            "Run the publication study before generating paper figures."
        )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    n_runs = 0
    for ax, bench in zip(axes.flat, benches):
        r = _plot_panel(ax, bench, dim)
        n_runs = r or n_runs
    for ax in axes[-1]:
        ax.set_xlabel("Iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("Global best fitness")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"Median convergence with IQR band across {n_runs} runs",
                 y=1.06, fontsize=12)
    fig.tight_layout()
    _savefig_all(fig, "convergence_grid")
    print("  wrote convergence_grid.{pdf,svg,png}")


def make_single(bench: str, dim: int) -> None:
    if (_load_histories("pso", bench, dim) is None
            or _load_histories("qpso_math", bench, dim) is None):
        raise RuntimeError(
            f"Missing convergence data for {bench} {dim}D. "
            "Run the publication study before generating paper figures."
        )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_runs = _plot_panel(ax, bench, dim)
    if not n_runs:
        print(f"  [skip] no data for {bench} {dim}D")
        plt.close(fig)
        return
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global best fitness")
    ax.legend(frameon=False)
    fig.tight_layout()
    _savefig_all(fig, f"convergence_{bench}")
    print(f"  wrote convergence_{bench}.{{pdf,svg,png}}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper convergence figures")
    p.add_argument("--dim", type=int, default=10,
                   help="Dimensionality for the convergence grid (default 10)")
    p.add_argument("--rosenbrock-dim", type=int, default=20,
                   help="Dimensionality for the Rosenbrock detail figure (default 20)")
    args = p.parse_args()

    print("Generating paper figures...")
    make_grid(args.dim)
    make_single("rosenbrock", args.rosenbrock_dim)
    print("Done.")


if __name__ == "__main__":
    main()
