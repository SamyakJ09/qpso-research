"""
Convergence utilities shared by PSO and QPSO.

Provides mean-best computation, worst-particle scattering for
stagnation recovery, and a StagnationDetector class.
"""

import numpy as np


def compute_mean_best(personal_bests: list[np.ndarray]) -> np.ndarray:
    """
    Mean Best Position (mbest) — average of all personal bests.

    This is the key anti-stagnation mechanism in QPSO:
      - Gives the swarm a shared reference point
      - Particles far from mbest explore more (wider quantum well)
      - Particles near mbest exploit more (tighter quantum well)
      - Prevents the entire swarm collapsing into one local region
    """
    return np.mean(personal_bests, axis=0)


def compute_weighted_mean_best(
    personal_bests: list[np.ndarray],
    best_scores: list[float],
) -> np.ndarray:
    """
    Weighted Mean Best Position (WQPSO) — fitness-weighted average of personal bests.

    Particles with better fitness contribute more to mbest, biasing the
    swarm reference toward higher-quality regions.  This is particularly
    effective on narrow-valley functions like Rosenbrock where the
    unweighted mean can be pulled away from the valley floor by distant,
    poorly-performing particles.

    Reference:
      Xi M, Sun J, Xu W. "An improved quantum-behaved particle swarm
      optimization algorithm with weighted mean best position."
      Applied Mathematics and Computation 205(2):1-15, 2008.
    """
    scores = np.array(best_scores)
    positions = np.array(personal_bests)

    # Invert scores so lower (better) fitness → higher weight.
    # Shift so the worst score maps to a small positive weight.
    worst = scores.max()
    best = scores.min()
    score_range = worst - best
    if score_range < 1e-30:
        # All particles have effectively equal fitness; fall back to uniform.
        return np.mean(positions, axis=0)

    # Weight: how far each particle's score is from the worst.
    weights = (worst - scores) + score_range * 0.01  # small floor to avoid zero
    weights /= weights.sum()

    return np.average(positions, axis=0, weights=weights)


def scatter_worst_particles(particles, global_best_pos: np.ndarray,
                            bounds: tuple[float, float], fraction: float = 0.3):
    """
    Convergence guarantee mechanism — re-scatter the worst-performing
    particles around the global best when stagnation is detected.

    Instead of staying stuck, the worst particles are teleported to
    random positions near the current best, creating new search paths
    without losing the best solution found so far.
    """
    lo, hi = bounds
    scores = [p.best_score for p in particles]
    n_scatter = max(1, int(len(particles) * fraction))
    worst_indices = np.argsort(scores)[-n_scatter:]

    spread = (hi - lo) * 0.15  # scatter within 15% of search range
    for idx in worst_indices:
        particles[idx].position = np.clip(
            global_best_pos + np.random.uniform(-spread, spread, len(global_best_pos)),
            lo, hi
        )
        particles[idx].best_score = float("inf")
        particles[idx].best_pos = particles[idx].position.copy()


class StagnationDetector:
    """
    Tracks whether the optimizer is stagnating (no improvement for N iterations).

    Parameters
    ----------
    limit : int
        Number of iterations without improvement before triggering stagnation.
    """

    def __init__(self, limit: int = 15):
        self.limit = limit
        self.count = 0
        self.last_best = float("inf")

    def update(self, current_best: float) -> bool:
        """
        Update with the current best score.

        Returns True if stagnation threshold has been reached (and resets counter).
        """
        if current_best < self.last_best - 1e-10:
            self.count = 0
            self.last_best = current_best
        else:
            self.count += 1

        if self.count >= self.limit:
            self.count = 0
            return True
        return False

    @property
    def is_stagnating(self) -> bool:
        """True if we're past half the stagnation limit (used for beta widening)."""
        return self.count >= self.limit // 2
