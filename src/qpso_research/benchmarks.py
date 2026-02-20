"""
Benchmark optimization functions for PSO vs QPSO comparison.

Contains 6 standard test functions used in swarm intelligence research,
plus a configurable shifted sphere for custom-goal experiments.
Each function has an associated BenchmarkInfo with bounds and known optima.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ─────────────────────────────────────────────
# Benchmark Functions
# ─────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    """Sphere function — global minimum at origin (f=0)."""
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function — highly multimodal, global minimum at origin (f=0)."""
    A = 10
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock (banana) function — global minimum at (1,1,...,1) (f=0)."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x: np.ndarray) -> float:
    """Ackley function — global minimum at origin (f=0). Bounds: [-32.768, 32.768]."""
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
                 - np.exp(sum_cos / n) + 20.0 + np.e)


def griewank(x: np.ndarray) -> float:
    """Griewank function — global minimum at origin (f=0). Bounds: [-600, 600]."""
    sum_sq = np.sum(x ** 2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_sq / 4000.0 - prod_cos + 1.0)


def schwefel(x: np.ndarray) -> float:
    """Schwefel function — global minimum at (420.9687,...) (f=0). Bounds: [-500, 500]."""
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def shifted_sphere(x: np.ndarray, goal: np.ndarray) -> float:
    """Shifted sphere — minimum is located exactly at goal coordinates (f=0)."""
    return float(np.sum((x - goal) ** 2))


# ─────────────────────────────────────────────
# Benchmark Registry
# ─────────────────────────────────────────────

@dataclass
class BenchmarkInfo:
    """Metadata for a benchmark function."""
    name: str
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]
    optimal_value: float

    def optimal_position(self, dims: int) -> np.ndarray:
        """Return the known optimal position for a given dimensionality."""
        return self._optimal_pos_fn(dims)

    _optimal_pos_fn: Callable[[int], np.ndarray] = None

    def __post_init__(self):
        # Set default optimal position function if not provided
        if self._optimal_pos_fn is None:
            self._optimal_pos_fn = lambda dims: np.zeros(dims)


BENCHMARKS: dict[str, BenchmarkInfo] = {
    "sphere": BenchmarkInfo(
        name="Sphere",
        func=sphere,
        bounds=(-10.0, 10.0),
        optimal_value=0.0,
        _optimal_pos_fn=lambda dims: np.zeros(dims),
    ),
    "rastrigin": BenchmarkInfo(
        name="Rastrigin",
        func=rastrigin,
        bounds=(-5.12, 5.12),
        optimal_value=0.0,
        _optimal_pos_fn=lambda dims: np.zeros(dims),
    ),
    "rosenbrock": BenchmarkInfo(
        name="Rosenbrock",
        func=rosenbrock,
        bounds=(-2.048, 2.048),
        optimal_value=0.0,
        _optimal_pos_fn=lambda dims: np.ones(dims),
    ),
    "ackley": BenchmarkInfo(
        name="Ackley",
        func=ackley,
        bounds=(-32.768, 32.768),
        optimal_value=0.0,
        _optimal_pos_fn=lambda dims: np.zeros(dims),
    ),
    "griewank": BenchmarkInfo(
        name="Griewank",
        func=griewank,
        bounds=(-600.0, 600.0),
        optimal_value=0.0,
        _optimal_pos_fn=lambda dims: np.zeros(dims),
    ),
    "schwefel": BenchmarkInfo(
        name="Schwefel",
        func=schwefel,
        bounds=(-500.0, 500.0),
        optimal_value=0.0,
        _optimal_pos_fn=lambda dims: np.full(dims, 420.9687),
    ),
}


def get_benchmark(name: str) -> BenchmarkInfo:
    """Look up a benchmark by name (case-insensitive)."""
    key = name.lower()
    if key not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARKS[key]
