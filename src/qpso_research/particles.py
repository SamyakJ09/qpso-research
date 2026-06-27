"""
Particle classes for PSO and QPSO.

PSOParticle uses velocity vectors (classical swarm dynamics).
QPSOParticle has no velocity — position is sampled from a quantum wave function.
"""

import numpy as np


class PSOParticle:
    """
    Classical PSO particle with position, velocity, and personal best tracking.

    Attributes
    ----------
    position    : current position in the search space
    velocity    : current velocity vector
    best_pos    : best position this particle has visited
    best_score  : fitness value at best_pos
    score       : current fitness value
    """

    def __init__(self, dimensions: int, bounds: tuple[float, float]):
        lo, hi = bounds
        self.position = np.random.uniform(lo, hi, dimensions)
        velocity_range = (hi - lo) * 0.1
        self.velocity = np.random.uniform(-velocity_range, velocity_range, dimensions)
        self.best_pos = self.position.copy()
        self.best_score = float("inf")
        self.score = float("inf")

    def evaluate(self, objective_fn) -> float:
        """Evaluate current position and update personal best."""
        self.score = objective_fn(self.position)
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_pos = self.position.copy()
        return self.score


class QPSOParticle:
    """
    Quantum PSO particle with optional wave-packet momentum.

    Position is determined by the quantum wave function mechanism:
    attractor point + quantum tunnelling displacement + momentum nudge.

    The momentum vector tracks the exponential moving average of recent
    position changes, modelling the de Broglie momentum component of the
    quantum wave packet ψ(x,t) = A·exp(i(kx − ωt)).  Standard QPSO only
    models |ψ|² (position distribution) but discards the momentum p = ℏk.
    Restoring it enables directional persistence for ridge-following on
    functions like Rosenbrock.
    """

    def __init__(self, dimensions: int, bounds: tuple[float, float]):
        lo, hi = bounds
        self.position = np.random.uniform(lo, hi, dimensions)
        self.prev_position = self.position.copy()
        self.momentum = np.zeros(dimensions)
        self.best_pos = self.position.copy()
        self.best_score = float("inf")
        self.score = float("inf")

    def evaluate(self, objective_fn) -> float:
        """Evaluate current position and update personal best."""
        self.score = objective_fn(self.position)
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_pos = self.position.copy()
        return self.score
