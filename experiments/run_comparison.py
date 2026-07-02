"""
Single-benchmark comparison: PSO vs QPSO.

Usage:
  qpso-compare --benchmark sphere --dims 2 --mode math
  qpso-compare --config configs/default.yaml --benchmark rastrigin --dims 5

  # Or with Python directly (use 'py' on Windows if 'python' fails):
  python experiments/run_comparison.py --benchmark sphere --dims 2 --mode math
  py -m qpso_research compare --benchmark sphere --dims 2
"""

from qpso_research.cli import run_comparison

if __name__ == "__main__":
    run_comparison()
