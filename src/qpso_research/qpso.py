"""
Unified Quantum Particle Swarm Optimization (QPSO).

Three modes via the `mode` parameter:
  - "math"   : pure numpy randomness (classical simulation of quantum behaviour)
  - "full"   : ALL randomness from Qiskit quantum circuits
  - "hybrid" : only tunnelling sign from Qiskit, rest classical

All modes share the same QPSO update equations:
  1. Compute attractor phi (weighted midpoint of personal + global best)
  2. Compute quantum well length L from mbest distance + Beta
  3. Sample displacement from quantum exponential distribution
  4. Apply tunnelling sign to land on either side of phi
  5. Clamp to search bounds

Sources:
  math   — pso_vs_qpsoTEST1.py
  full   — pso_vs_qpsoTEST2.py
  hybrid — pso_vs_qpsoTEST3.py
"""

import time

import numpy as np

from .particles import QPSOParticle
from .convergence import compute_mean_best, scatter_worst_particles, StagnationDetector


class QPSO:
    """
    Quantum Particle Swarm Optimizer with configurable randomness source.

    Parameters
    ----------
    objective_fn      : callable — function to minimise
    dimensions        : int — number of decision variables
    bounds            : (float, float) — search space limits
    num_particles     : int — swarm size
    max_iterations    : int — stopping criterion
    beta_start        : float — initial quantum well size (broad exploration)
    beta_end          : float — final quantum well size (tight exploitation)
    cognitive_coef    : float — c1, personal-best attraction weight
    social_coef       : float — c2, global-best attraction weight
    stagnation_limit  : int — iterations without improvement before scatter
    success_threshold : float — score below this counts as "reached goal"
    mode              : str — "math", "full", or "hybrid"
    shots             : int — Qiskit shots per circuit (only for full/hybrid modes)
    """

    VALID_MODES = ("math", "full", "hybrid")

    def __init__(
        self,
        objective_fn,
        dimensions: int,
        bounds: tuple[float, float],
        num_particles: int = 30,
        max_iterations: int = 150,
        beta_start: float = 1.0,
        beta_end: float = 0.5,
        cognitive_coef: float = 1.494,
        social_coef: float = 1.494,
        stagnation_limit: int = 15,
        success_threshold: float = 1e-6,
        mode: str = "math",
        shots: int = 256,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.fn = objective_fn
        self.dims = dimensions
        self.bounds = bounds
        self.n = num_particles
        self.max_iter = max_iterations
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.c1 = cognitive_coef
        self.c2 = social_coef
        self.threshold = success_threshold
        self.mode = mode

        self.stagnation = StagnationDetector(limit=stagnation_limit)

        # Lazy-init quantum engine only when needed
        self.qengine = None
        if mode in ("full", "hybrid"):
            from .quantum_engine import QuantumEngine
            self.qengine = QuantumEngine(shots=shots)

        # State
        self.swarm: list[QPSOParticle] = []
        self.global_best_pos: np.ndarray | None = None
        self.global_best_score: float = float("inf")

        # History
        self.best_history: list[float] = []
        self.avg_history: list[float] = []
        self.mbest_history: list[float] = []
        self.beta_history: list[float] = []
        self.iter_to_success: int | None = None
        self.elapsed: float = 0.0

    def _beta(self, iteration: int, stagnating: bool = False) -> float:
        """Compute Beta with linear decay and stagnation widening."""
        progress = (iteration - 1) / max(self.max_iter - 1, 1)
        beta = self.beta_start - progress * (self.beta_start - self.beta_end)
        if stagnating:
            beta = min(beta * 1.3, self.beta_start)  # widen well to escape
        return beta

    def _get_random_values(self, beta: float) -> tuple:
        """
        Get random values for position update based on mode.

        Returns (r1_all, r2_all, u_all, sign_all) each of shape (n_particles, dims).
        """
        if self.mode == "math":
            r1_all = np.random.random((self.n, self.dims))
            r2_all = np.random.random((self.n, self.dims))
            u_all = np.random.uniform(0.001, 0.999, (self.n, self.dims))
            sign_all = np.where(
                np.random.random((self.n, self.dims)) < 0.5, 1.0, -1.0
            )
            return r1_all, r2_all, u_all, sign_all

        elif self.mode == "full":
            return self.qengine.sample_batch_full(self.n, self.dims, beta)

        else:  # hybrid
            r1_all = np.random.random((self.n, self.dims))
            r2_all = np.random.random((self.n, self.dims))
            u_all = np.random.uniform(0.001, 0.999, (self.n, self.dims))
            sign_all = self.qengine.sample_signs_only(self.n, self.dims, beta)
            return r1_all, r2_all, u_all, sign_all

    def optimize(self, verbose: bool = True, goal_coords: list | None = None) -> dict:
        """
        Run the QPSO optimization loop.

        Parameters
        ----------
        verbose      : print progress every 25 iterations
        goal_coords  : optional known goal position for mbest distance tracking

        Returns
        -------
        dict with keys: best_position, best_score, best_history, avg_history,
                        mbest_history, beta_history, iter_to_success, elapsed_time
        """
        lo, hi = self.bounds
        self.swarm = [QPSOParticle(self.dims, self.bounds) for _ in range(self.n)]
        start_time = time.time()

        if verbose and self.mode != "math" and self.qengine is not None:
            print(f"  [QPSO] Mode            : {self.mode}")
            print(f"  [QPSO] Qiskit backend  : {self.qengine.backend.name}")
            print(f"  [QPSO] Shots/circuit   : {self.qengine.shots}")

        for it in range(1, self.max_iter + 1):
            stagnating = self.stagnation.is_stagnating
            beta = self._beta(it, stagnating)
            fine_tuning = it > self.max_iter * 0.9
            self.beta_history.append(beta)

            # ── Evaluate ──────────────────────────────────────────
            scores = []
            for p in self.swarm:
                score = p.evaluate(self.fn)
                scores.append(score)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos = p.position.copy()

            # ── Mean best ─────────────────────────────────────────
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

            # ── Get random values (mode-dependent) ────────────────
            r1_all, r2_all, u_all, sign_all = self._get_random_values(beta)

            # ── Quantum position update ───────────────────────────
            for i, p in enumerate(self.swarm):
                r1 = r1_all[i]
                r2 = r2_all[i]
                u = u_all[i]
                sign = sign_all[i]

                # Attractor — weighted midpoint of personal + global best
                phi = (self.c1 * r1 * p.best_pos + self.c2 * r2 * self.global_best_pos) / \
                      (self.c1 * r1 + self.c2 * r2 + 1e-10)

                if fine_tuning:
                    tight_beta = max(beta * 0.3, 0.1)
                    L = (2.0 / tight_beta) * np.abs(phi - self.global_best_pos) + 1e-8
                else:
                    L = (2.0 / beta) * np.abs(phi - mbest) + 1e-8

                delta = (L / 2.0) * np.log(1.0 / u)
                p.position = np.clip(phi + sign * delta, lo, hi)

            if verbose and (it % 25 == 0 or it == 1):
                print(f"  [QPSO] Iter {it:>4d}/{self.max_iter} | "
                      f"Beta={beta:.3f} | Best: {self.global_best_score:.2e} | "
                      f"mbest_dist: {mbest_dist:.4f}")

        self.elapsed = time.time() - start_time

        return {
            "best_position": self.global_best_pos,
            "best_score": self.global_best_score,
            "best_history": self.best_history,
            "avg_history": self.avg_history,
            "mbest_history": self.mbest_history,
            "beta_history": self.beta_history,
            "iter_to_success": self.iter_to_success,
            "elapsed_time": self.elapsed,
        }
