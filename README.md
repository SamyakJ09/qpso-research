# PSO vs QPSO Research

**How can Quantum Particle Swarm Optimization improve efficiency in swarm coordination?**

A publication-quality comparison framework for Classical PSO vs Quantum PSO (QPSO) with three quantum randomness modes.

## QPSO Modes

| Mode | Randomness Source | Description |
|------|------------------|-------------|
| `math` | numpy (classical) | Mathematical simulation of quantum behaviour |
| `full` | Qiskit circuits | ALL random values from quantum circuit measurements |
| `hybrid` | Qiskit + numpy | Only tunnelling sign from quantum, rest classical |

## Quick Start

```bash
# One-command setup (creates .venv, installs everything)
bash setup.sh          # Linux/macOS/Git Bash/Windows (Git Bash)

# Or manually:
py -m venv .venv                 # Windows ('py' is the official launcher — see PEP 397)
python3 -m venv .venv            # Linux/macOS
source .venv/Scripts/activate    # Windows (Git Bash)
source .venv/bin/activate        # Linux/macOS
pip install -e ".[dev]"

# For quantum modes (full/hybrid):
pip install -e ".[quantum]"
```

## Running Experiments

### Which config do I run?

| Config | Scope | Runs / iters | Runtime | Use for |
|--------|-------|--------------|---------|---------|
| `configs/default.yaml` | 3 benchmarks, dims 2 & 5 | 5 runs, 150 iters | seconds | Quick smoke test only — **not** for reporting (too few runs/iters for significance) |
| `configs/rastrigin_powered.yaml` | Rastrigin, dims 2/5/10/20 | 50 runs, 500 iters, `math` | ~2.5 min | Properly-powered Rastrigin sweep — the multimodal result |
| `configs/publication.yaml` | 6 benchmarks, dims 2/5/10/20 | 50 runs, 500 iters, `math` | ~30 min | **The paper study** — fills the scaling + significance tables |

> **Reporting numbers:** always use a 50-run config. `default.yaml` cannot produce significant results
> even when a real effect exists. For Rastrigin specifically, report **success/escape rate**
> (fraction of runs reaching the global optimum / basin), not just mean fitness — the mean is
> dominated by run-to-run variance and hides QPSO's local-minimum-escape advantage.
>
> **Quantum modes (`full`/`hybrid`):** the circuit-based randomness modes require
> `pip install -e ".[quantum]"` and are ~100× slower than `math` (a Qiskit circuit is built every
> iteration). They are **not** part of the main study grid — the empirical paper uses `math` mode.
> A `math`-vs-`hybrid`-vs-`full` comparison is left as a separate, reduced sub-study (future work).

### Commands

After installation, three equivalent methods are available — choose whichever works on your system:

```bash
# Method 1 — Installed commands (recommended, works everywhere)
qpso-compare --benchmark rastrigin --dims 5 --mode math   # single PSO-vs-QPSO run (illustrative)
qpso-study --config configs/rastrigin_powered.yaml        # powered Rastrigin sweep (~2.5 min)
qpso-study --config configs/publication.yaml              # full paper study, math mode (~30 min)

# Method 2 — Module execution (works with 'py' on Windows, 'python3' on Unix)
py -m qpso_research compare --benchmark rastrigin --dims 5
py -m qpso_research study --config configs/rastrigin_powered.yaml

# Method 3 — Script execution
python experiments/run_comparison.py --benchmark rastrigin --dims 5 --mode math
python experiments/run_full_study.py --config configs/rastrigin_powered.yaml

# Run tests
pytest -v
```

Outputs land in `results/`: raw per-run CSVs in `results/raw/`, boxplots in `results/figures/`,
and the summary table + `experiment_summary.json` (means, stds, Wilcoxon p-values) in `results/summary/`.

## Reproducing the Paper

The paper (`paper/`) is built from the study outputs in three steps. Run them from the repo root
after the study has populated `results/`:

```bash
# 1. Generate the data (writes results/raw, results/figures, results/summary)
qpso-study --config configs/publication.yaml          # ~30 min, math mode, 50 runs

# 2. Generate the convergence figures referenced by the paper
python experiments/make_paper_figures.py              # -> results/figures/convergence_*.png
#   options: --dim N (grid dimensionality, default 10), --rosenbrock-dim N (default 20)

# 3. Build the PDF (requires a LaTeX toolchain with biber; e.g. MiKTeX or TeX Live)
cd paper && latexmk -pdf main.tex                     # -> paper/build/main.pdf
```

Figures are read from `results/figures/` via `\graphicspath` in `paper/preamble.tex`, so the study
and figure steps must run before compiling. The boxplots (`boxplot_math_<fn>_<dim>d.png`) are produced
directly by the study; the median-convergence curves come from step 2.

> **Data ↔ paper consistency:** the numbers in `paper/sections/results.tex` (Tables — scaling,
> Wilcoxon, and Rastrigin escape-rate) are transcribed from a completed `publication.yaml` run plus
> the Rastrigin escape-rate analysis. If you re-run the study with different settings, regenerate the
> figures and update those tables to match `results/summary/`.

## Windows Troubleshooting

If you see *"Python was not found; run without arguments to install from the Microsoft Store"*:

Windows 10/11 ships with app-execution aliases that intercept the bare `python` command and redirect to the Microsoft Store. Three fixes, from easiest to most permanent:

1. **Use `py` instead of `python`** — The [Python Launcher](https://peps.python.org/pep-0397/) (`py`) ships with every python.org install and is immune to the Store alias. The setup scripts and all documented commands use `py` by default.
2. **Disable the alias permanently** — Open **Settings > Apps > Advanced app settings > App execution aliases** and turn OFF `python.exe` and `python3.exe`.
3. **Use the installed commands** — After `pip install -e ".[dev]"`, use `qpso-compare` and `qpso-study` directly. These are standalone executables that bypass `python` entirely.

## Project Structure

```
src/qpso_research/       Core library
  benchmarks.py          6 benchmark functions + registry
  particles.py           PSOParticle + QPSOParticle
  pso.py                 Classical PSO with convergence guarantees
  qpso.py                Unified QPSO (mode=math|full|hybrid)
  quantum_engine.py      Qiskit circuit engine (lazy import)
  convergence.py         mbest, stagnation detection, scatter
  config.py              YAML config loader
  cli.py                 Console entry points (qpso-compare, qpso-study)
  __main__.py            python -m qpso_research support
  statistics.py          Wilcoxon + Friedman tests
  visualization.py       4-panel plots, convergence curves, boxplots
  results.py             CSV/JSON persistence

experiments/             Thin wrappers (import from cli.py)
  run_comparison.py      Single-benchmark PSO vs QPSO
  run_full_study.py      Multi-benchmark statistical study
  make_paper_figures.py  Median+IQR convergence figures for the paper

configs/                 YAML configuration files
  default.yaml           Quick smoke test (3 benchmarks, 5 runs, 150 iters)
  rastrigin_powered.yaml Powered Rastrigin sweep (dims 2/5/10/20, 50 runs, 500 iters)
  publication.yaml       Paper study (6 benchmarks, dims 2/5/10/20, 50 runs, math mode)

paper/                   LaTeX manuscript (build with: cd paper && latexmk -pdf main.tex)
  main.tex               Document root; sections/ holds each section
  build/main.pdf         Compiled output

tests/                   Unit tests
results/                 Auto-generated output (gitignored)
```

## Benchmark Functions

| Function | Bounds | Optimum | Characteristics |
|----------|--------|---------|-----------------|
| Sphere | [-10, 10] | f(0,...,0) = 0 | Unimodal, smooth |
| Rastrigin | [-5.12, 5.12] | f(0,...,0) = 0 | Highly multimodal |
| Rosenbrock | [-2.048, 2.048] | f(1,...,1) = 0 | Narrow valley |
| Ackley | [-32.768, 32.768] | f(0,...,0) = 0 | Multimodal, flat regions |
| Griewank | [-600, 600] | f(0,...,0) = 0 | Many local minima |
| Schwefel | [-500, 500] | f(420.97,...) = 0 | Deceptive global minimum |

## Key Mechanisms

- **Weighted Mean Best (WQPSO)**: Fitness-weighted average of personal bests, biases swarm toward better regions
- **Wave-packet Momentum**: De Broglie-inspired directional persistence for ridge-following
- **Stagnation Detection**: Worst 30% of particles scattered when no improvement detected
- **Elitist Refinement**: Differential + Gaussian local search on global best
- **Beta Scheduling**: Linear decay from exploration (1.0) to exploitation (0.5) with stagnation widening

## License

MIT
