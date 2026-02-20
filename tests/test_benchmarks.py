"""Tests for benchmark functions."""

import numpy as np
import pytest

from qpso_research.benchmarks import (
    sphere, rastrigin, rosenbrock, ackley, griewank, schwefel,
    shifted_sphere, get_benchmark, BENCHMARKS,
)


class TestBenchmarkValues:
    """Verify each function returns correct values at known optima."""

    def test_sphere_at_origin(self):
        assert sphere(np.zeros(5)) == pytest.approx(0.0, abs=1e-10)

    def test_sphere_nonzero(self):
        assert sphere(np.array([1.0, 2.0, 3.0])) == pytest.approx(14.0)

    def test_rastrigin_at_origin(self):
        assert rastrigin(np.zeros(3)) == pytest.approx(0.0, abs=1e-10)

    def test_rastrigin_multimodal(self):
        # Should be > 0 away from origin
        assert rastrigin(np.array([1.0, 1.0])) > 0

    def test_rosenbrock_at_ones(self):
        assert rosenbrock(np.ones(4)) == pytest.approx(0.0, abs=1e-10)

    def test_rosenbrock_away_from_optimum(self):
        assert rosenbrock(np.zeros(3)) > 0

    def test_ackley_at_origin(self):
        assert ackley(np.zeros(5)) == pytest.approx(0.0, abs=1e-10)

    def test_ackley_positive_away(self):
        assert ackley(np.array([1.0, 2.0])) > 0

    def test_griewank_at_origin(self):
        assert griewank(np.zeros(3)) == pytest.approx(0.0, abs=1e-10)

    def test_schwefel_at_optimum(self):
        opt = np.full(2, 420.9687)
        assert schwefel(opt) == pytest.approx(0.0, abs=0.1)

    def test_shifted_sphere(self):
        goal = np.array([3.0, -5.0, 2.0])
        assert shifted_sphere(goal, goal) == pytest.approx(0.0, abs=1e-10)
        assert shifted_sphere(np.zeros(3), goal) > 0


class TestRegistry:
    """Verify the benchmark registry works correctly."""

    def test_all_benchmarks_registered(self):
        expected = {"sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "schwefel"}
        assert set(BENCHMARKS.keys()) == expected

    def test_get_benchmark_case_insensitive(self):
        bench = get_benchmark("SPHERE")
        assert bench.name == "Sphere"

    def test_get_benchmark_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent")

    def test_optimal_position_shape(self):
        for name, bench in BENCHMARKS.items():
            pos = bench.optimal_position(5)
            assert pos.shape == (5,), f"{name} optimal_position wrong shape"
