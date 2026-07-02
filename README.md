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

After installation, three equivalent methods are available — choose whichever works on your system:

```bash
# Method 1 — Installed commands (recommended, works everywhere)
qpso-compare --benchmark sphere --dims 2 --mode math
qpso-study --config configs/default.yaml

# Method 2 — Module execution (works with 'py' on Windows, 'python3' on Unix)
py -m qpso_research compare --benchmark sphere --dims 2
py -m qpso_research study --config configs/default.yaml

# Method 3 — Script execution
python experiments/run_comparison.py --benchmark sphere --dims 2 --mode math
python experiments/run_full_study.py --config configs/default.yaml

# Run tests
pytest -v
```

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

configs/                 YAML configuration files
  default.yaml           Quick test (3 benchmarks, 5 runs)
  publication.yaml       Full study (6 benchmarks, 30 runs)

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
