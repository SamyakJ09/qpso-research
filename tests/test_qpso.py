"""Tests for QPSO optimizer."""

import numpy as np
import pytest

from qpso_research.benchmarks import sphere
from qpso_research.qpso import QPSO


class TestQPSO:
    """Verify QPSO math mode converges and returns correct result format."""

    def test_returns_dict(self):
        np.random.seed(42)
        qpso = QPSO(sphere, dimensions=2, bounds=(-10, 10),
                     max_iterations=10, num_particles=5, mode="math")
        result = qpso.optimize(verbose=False)
        assert isinstance(result, dict)
        assert "best_position" in result
        assert "best_score" in result
        assert "beta_history" in result

    def test_converges_on_sphere(self):
        np.random.seed(42)
        qpso = QPSO(
            sphere,
            dimensions=2,
            bounds=(-10, 10),
            num_particles=30,
            max_iterations=100,
            mode="math",
        )
        result = qpso.optimize(verbose=False)
        assert result["best_score"] < 1e-3, "QPSO math mode should converge on sphere"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            QPSO(sphere, dimensions=2, bounds=(-10, 10), mode="invalid")

    def test_history_length(self):
        np.random.seed(42)
        max_iter = 50
        qpso = QPSO(sphere, dimensions=2, bounds=(-10, 10),
                     max_iterations=max_iter, num_particles=5, mode="math")
        result = qpso.optimize(verbose=False)
        assert len(result["best_history"]) == max_iter
        assert len(result["beta_history"]) == max_iter

    def test_beta_history_decreasing_trend(self):
        np.random.seed(42)
        qpso = QPSO(sphere, dimensions=2, bounds=(-10, 10),
                     max_iterations=50, num_particles=5, mode="math",
                     beta_start=1.0, beta_end=0.5)
        result = qpso.optimize(verbose=False)
        betas = result["beta_history"]
        # Overall trend should be decreasing (first > last)
        assert betas[0] >= betas[-1]

    def test_best_history_monotonic(self):
        np.random.seed(42)
        qpso = QPSO(sphere, dimensions=2, bounds=(-10, 10),
                     max_iterations=50, num_particles=10, mode="math")
        result = qpso.optimize(verbose=False)
        history = result["best_history"]
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1], "best_history should be monotonically decreasing"
