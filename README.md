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
bash setup.sh          # Linux/macOS/Git Bash
setup.bat              # Windows CMD

# Or manually:
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
source .venv/bin/activate       # Linux/macOS
pip install -e ".[dev]"

# For quantum modes (full/hybrid):
pip install -e ".[quantum]"

# Run a single comparison
python experiments/run_comparison.py --benchmark sphere --dims 2 --mode math

# Run with a config file
python experiments/run_comparison.py --config configs/default.yaml --benchmark rastrigin --dims 5

# Run the full study
python experiments/run_full_study.py --config configs/default.yaml

# Run tests
pytest -v
```

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
  statistics.py          Wilcoxon + Friedman tests
  visualization.py       4-panel plots, convergence curves, boxplots
  results.py             CSV/JSON persistence

experiments/             Experiment runners
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

- **Mean Best Position (mbest)**: Average of all personal bests, used as quantum reference centre
- **Stagnation Detection**: Worst 30% of particles scattered when no improvement detected
- **Fine-tuning Phase**: Final 10% of iterations tighten search around best known position
- **Beta Scheduling**: Linear decay from exploration (1.0) to exploitation (0.5) with stagnation widening

## License

MIT
