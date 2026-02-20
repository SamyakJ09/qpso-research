"""
Results persistence — save and load experiment data.

Handles CSV files for per-run convergence histories and
JSON files for aggregated experiment summaries.
"""

import json
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def save_run_result(result: dict, filepath: str | Path) -> None:
    """
    Save a single run's convergence history to CSV.

    Columns: iteration, best_score, avg_score, mbest_dist
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n = len(result["best_history"])
    with open(filepath, "w") as f:
        f.write("iteration,best_score,avg_score,mbest_dist\n")
        for i in range(n):
            best = result["best_history"][i]
            avg = result["avg_history"][i]
            mbest = result["mbest_history"][i] if i < len(result["mbest_history"]) else ""
            f.write(f"{i + 1},{best},{avg},{mbest}\n")


def save_experiment_summary(summary: dict, filepath: str | Path) -> None:
    """Save aggregated experiment results to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)


def load_experiment_summary(filepath: str | Path) -> dict:
    """Load experiment summary from JSON."""
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        return json.load(f)
