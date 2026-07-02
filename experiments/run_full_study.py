"""
Full publication study: multi-benchmark, multi-dimension, multi-run comparison.

Usage:
  qpso-study --config configs/default.yaml
  qpso-study --config configs/publication.yaml

  # Or with Python directly (use 'py' on Windows if 'python' fails):
  python experiments/run_full_study.py --config configs/default.yaml
  py -m qpso_research study --config configs/default.yaml
"""

from qpso_research.cli import run_full_study

if __name__ == "__main__":
    run_full_study()
