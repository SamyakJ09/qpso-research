"""Tests for PSO optimizer."""

import numpy as np
import pytest

from qpso_research.benchmarks import sphere
from qpso_research.pso import PSO


class TestPSO:
    """Verify PSO converges and returns correct result format."""

    def test_returns_dict(self):
        np.random.seed(42)
        pso = PSO(sphere, dimensions=2, bounds=(-10, 10), max_iterations=10, num_particles=5)
        result = pso.optimize(verbose=False)
        assert isinstance(result, dict)
        assert "best_position" in result
        assert "best_score" in result
        assert "best_history" in result
        assert "elapsed_time" in result

    def test_converges_on_sphere(self):
        np.random.seed(42)
        pso = PSO(
            sphere,
            dimensions=2,
            bounds=(-10, 10),
            num_particles=30,
            max_iterations=100,
        )
        result = pso.optimize(verbose=False)
        assert result["best_score"] < 1e-3, "PSO should converge close to 0 on sphere"

    def test_history_length(self):
        np.random.seed(42)
        max_iter = 50
        pso = PSO(sphere, dimensions=2, bounds=(-10, 10),
                  max_iterations=max_iter, num_particles=5)
        result = pso.optimize(verbose=False)
        assert len(result["best_history"]) == max_iter
        assert len(result["avg_history"]) == max_iter

    def test_best_history_monotonic(self):
        np.random.seed(42)
        pso = PSO(sphere, dimensions=2, bounds=(-10, 10),
                  max_iterations=50, num_particles=10)
        result = pso.optimize(verbose=False)
        history = result["best_history"]
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1], "best_history should be monotonically decreasing"

    def test_iter_to_success_tracked(self):
        np.random.seed(42)
        pso = PSO(
            sphere,
            dimensions=2,
            bounds=(-5, 5),
            num_particles=30,
            max_iterations=200,
            success_threshold=1e-4,
        )
        result = pso.optimize(verbose=False)
        if result["best_score"] < 1e-4:
            assert result["iter_to_success"] is not None
            assert result["iter_to_success"] <= 200
