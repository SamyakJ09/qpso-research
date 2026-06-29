"""
Classical Particle Swarm Optimization (PSO).

Single definitive PSO implementation with:
  - Adaptive inertia scheduling (linear decay)
  - Velocity clamping
  - Mean best position (mbest) tracking
  - Stagnation detection with worst-particle scatter
  - Fine-tuning phase in final 10% of iterations
  - Convergence guarantee mechanisms

Based on Clerc & Kennedy (2002) constriction coefficient PSO with
convergence guarantees via stagnation detection and particle scattering.
"""

import time

import numpy as np

from .particles import PSOParticle
from .convergence import compute_mean_best, scatter_worst_particles, StagnationDetector


class PSO:
    """
    Classical Particle Swarm Optimizer.

    Parameters
    ----------
    objective_fn      : callable — function to minimise
    dimensions        : int — number of decision variables
    bounds            : (float, float) — search space limits
    num_particles     : int — swarm size
    max_iterations    : int — stopping criterion
    inertia_start     : float — initial inertia weight
    inertia_end       : float — final inertia weight
    cognitive_coef    : float — c1, personal-best attraction
    social_coef       : float — c2, global-best attraction
    velocity_clamp    : float — max speed as fraction of search range
    stagnation_limit  : int — iterations without improvement before scatter
    success_threshold : float — score below this counts as "reached goal"
    """

    def __init__(
        self,
        objective_fn,
        dimensions: int,
        bounds: tuple[float, float],
        num_particles: int = 30,
        max_iterations: int = 150,
        inertia_start: float = 0.9,
        inertia_end: float = 0.4,
        cognitive_coef: float = 1.494,
        social_coef: float = 1.494,
        velocity_clamp: float = 0.2,
        stagnation_limit: int = 15,
        success_threshold: float = 1e-6,
    ):
        self.fn = objective_fn
        self.dims = dimensions
        self.bounds = bounds
        self.n = num_particles
        self.max_iter = max_iterations
        self.w_start = inertia_start
        self.w_end = inertia_end
        self.c1 = cognitive_coef
        self.c2 = social_coef
        self.v_max = velocity_clamp * (bounds[1] - bounds[0])
        self.threshold = success_threshold

        self.stagnation = StagnationDetector(limit=stagnation_limit)

        # State
        self.swarm: list[PSOParticle] = []
        self.global_best_pos: np.ndarray | None = None
        self.global_best_score: float = float("inf")

        # History
        self.best_history: list[float] = []
        self.avg_history: list[float] = []
        self.mbest_history: list[float] = []
        self.iter_to_success: int | None = None
        self.elapsed: float = 0.0

    def _inertia(self, iteration: int) -> float:
        """Linear inertia decay from w_start to w_end."""
        progress = (iteration - 1) / max(self.max_iter - 1, 1)
        return self.w_start - progress * (self.w_start - self.w_end)

    def optimize(self, verbose: bool = True, goal_coords: list | None = None) -> dict:
        """
        Run the PSO optimization loop.

        Parameters
        ----------
        verbose      : print progress every 25 iterations
        goal_coords  : optional known goal position for mbest distance tracking

        Returns
        -------
        dict with keys: best_position, best_score, best_history, avg_history,
                        mbest_history, iter_to_success, elapsed_time
        """
        lo, hi = self.bounds
        self.swarm = [PSOParticle(self.dims, self.bounds) for _ in range(self.n)]
        start_time = time.time()

        for it in range(1, self.max_iter + 1):
            w = self._inertia(it)
            fine_tuning = it > self.max_iter * 0.9

            # ── Evaluate ─────────────────────────────────────────
            scores = []
            for p in self.swarm:
                score = p.evaluate(self.fn)
                scores.append(score)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos = p.position.copy()

            # ── Mean best position ────────────────────────────────
            mbest = compute_mean_best([p.best_pos for p in self.swarm])
            if goal_coords is not None:
                mbest_dist = float(np.linalg.norm(mbest - np.array(goal_coords)))
            else:
                mbest_dist = float(np.linalg.norm(mbest - self.global_best_pos))
            self.mbest_history.append(mbest_dist)

            self.best_history.append(self.global_best_score)
            self.avg_history.append(float(np.mean(scores)))

            # ── Success check ─────────────────────────────────────
            if self.global_best_score <= self.threshold and self.iter_to_success is None:
                self.iter_to_success = it

            # ── Stagnation detection ──────────────────────────────
            if self.stagnation.update(self.global_best_score):
                scatter_worst_particles(self.swarm, self.global_best_pos, self.bounds)

            # ── Velocity & position update ────────────────────────
            for p in self.swarm:
                r1 = np.random.random(self.dims)
                r2 = np.random.random(self.dims)

                if fine_tuning:
                    p.velocity = (0.3 * p.velocity
                                  + 2.5 * r1 * (p.best_pos - p.position)
                                  + 2.5 * r2 * (self.global_best_pos - p.position))
                else:
                    p.velocity = (w * p.velocity
                                  + self.c1 * r1 * (p.best_pos - p.position)
                                  + self.c2 * r2 * (self.global_best_pos - p.position))

                p.velocity = np.clip(p.velocity, -self.v_max, self.v_max)
                p.position = np.clip(p.position + p.velocity, lo, hi)

            if verbose and (it % 25 == 0 or it == 1):
                print(f"  [PSO]  Iter {it:>4d}/{self.max_iter} | "
                      f"w={w:.3f} | Best: {self.global_best_score:.2e} | "
                      f"mbest_dist: {mbest_dist:.4f}")

        self.elapsed = time.time() - start_time

        return {
            "best_position": self.global_best_pos,
            "best_score": self.global_best_score,
            "best_history": self.best_history,
            "avg_history": self.avg_history,
            "mbest_history": self.mbest_history,
            "iter_to_success": self.iter_to_success,
            "elapsed_time": self.elapsed,
        }
