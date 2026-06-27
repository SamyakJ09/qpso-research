"""
Unified Quantum Particle Swarm Optimization (QPSO).

Three modes via the `mode` parameter:
  - "math"   : pure numpy randomness (classical simulation of quantum behaviour)
  - "full"   : ALL randomness from Qiskit quantum circuits
  - "hybrid" : only tunnelling sign from Qiskit, rest classical

Implements the canonical QPSO position update (Sun et al. 2004, 2012)
enhanced with two literature-backed improvements:

  1. Compute attractor p_i = phi * pbest + (1-phi) * gbest
  2. Compute displacement = beta * |mbest - x_i| * ln(1/u)
  3. Apply tunnelling sign: x_quantum = p_i ± displacement
  4. Apply wave-packet momentum: x_i = x_quantum + gamma * momentum_i
  5. Clamp to search bounds

Enhancements:
  - Weighted Mean Best (WQPSO, Xi et al. 2008): fitness-weighted mbest
    biases the swarm reference toward better solutions.
  - Wave-packet momentum (de Broglie): each particle tracks an EMA of
    its recent position changes.  Standard QPSO models only the position
    distribution |ψ|² but discards the momentum p = ℏk of the wave
    function ψ = A·exp(i(kx − ωt)).  Restoring it enables directional
    persistence for following narrow curved valleys (Rosenbrock).
  - Elitist refinement (EB-QPSO): differential direction step +
    coordinate-wise Gaussian perturbation of the global best.

Sources:
  Sun J, Xu W, Feng B. "A global search strategy of quantum-behaved
    particle swarm optimization." IEEE CCC, 2004.
  Sun J, Fang W, Wu X, Palade V, Xu W. "Quantum-behaved particle swarm
    optimization: analysis of individual particle behavior and parameter
    selection." Evolutionary Computation 20(3):349-393, 2012.
  Xi M, Sun J, Xu W. "An improved quantum-behaved particle swarm
    optimization algorithm with weighted mean best position."
    Applied Mathematics and Computation, 2008.
  Fallahi S, Taghadosi M. "Quantum-behaved particle swarm optimization
    based on solitons." Scientific Reports 12:13977, 2022.
"""

import time

import numpy as np

from .particles import QPSOParticle
from .convergence import (
    compute_mean_best,
    compute_weighted_mean_best,
    scatter_worst_particles,
    StagnationDetector,
)


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
    momentum_decay    : float — EMA decay for wave-packet momentum (0 = off, 0.8 = strong)
    momentum_weight   : float — strength of momentum nudge on position update
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
        momentum_decay: float = 0.7,
        momentum_weight: float = 0.1,
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
        self.mom_decay = momentum_decay
        self.mom_weight = momentum_weight

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

    def _refine_global_best(self, beta: float) -> None:
        """
        Elitist refinement of the global best using two strategies:

        1. Differential mutation — perturb gbest along a direction derived
           from two random particles' personal bests.  On valley functions
           like Rosenbrock, particles spread along the ridge, so the
           differential vector naturally aligns with the valley curve.
        2. Coordinate-wise Gaussian — try a small random step in each
           dimension independently (EB-QPSO style).

        Both strategies are greedy: improvements are kept immediately.
        """
        lo, hi = self.bounds

        # Strategy 1: differential direction step (DE-inspired)
        # Uses direction from two random particles' bests, scaled to
        # a controlled step size.  On valley functions, particles spread
        # along the ridge, so the direction naturally follows the curve.
        if len(self.swarm) >= 3:
            idxs = np.random.choice(len(self.swarm), 2, replace=False)
            diff = self.swarm[idxs[0]].best_pos - self.swarm[idxs[1]].best_pos
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                sigma_dir = beta * (hi - lo) * 0.02
                direction = diff / norm
                candidate = np.clip(
                    self.global_best_pos + sigma_dir * direction, lo, hi
                )
                score = self.fn(candidate)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos = candidate.copy()

        # Strategy 2: coordinate-wise Gaussian perturbation
        candidate = self.global_best_pos.copy()
        sigma = beta * (hi - lo) * 0.02
        for d in range(self.dims):
            trial = candidate.copy()
            trial[d] += np.random.normal(0, sigma)
            trial[d] = np.clip(trial[d], lo, hi)
            score = self.fn(trial)
            if score < self.global_best_score:
                candidate[d] = trial[d]
                self.global_best_score = score
                self.global_best_pos = candidate.copy()

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
            self.beta_history.append(beta)

            # ── Evaluate ──────────────────────────────────────────
            scores = []
            for p in self.swarm:
                score = p.evaluate(self.fn)
                scores.append(score)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos = p.position.copy()

            # ── Weighted mean best (WQPSO) ────────────────────────
            # Fitness-weighted mbest biases toward better particles,
            # pulling the swarm reference toward the valley floor on
            # narrow-valley functions like Rosenbrock.
            mbest = compute_weighted_mean_best(
                [p.best_pos for p in self.swarm],
                [p.best_score for p in self.swarm],
            )
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

            # ── Elitist local refinement (every 5 iterations) ─────
            if it % 5 == 0:
                self._refine_global_best(beta)

            # ── Get random values (mode-dependent) ────────────────
            _, _, u_all, sign_all = self._get_random_values(beta)

            # ── Quantum position update (canonical Sun et al.) ────
            # Enhanced with de Broglie wave-packet momentum: the
            # particle's wave function ψ = A·exp(i(kx-ωt)) carries
            # both a position distribution (|ψ|², modelled by the
            # standard quantum step) and a momentum p = ℏk (modelled
            # by the EMA of recent position changes).
            for i, p in enumerate(self.swarm):
                u = u_all[i]
                sign = sign_all[i]
                prev_pos = p.position.copy()

                # Attractor — random interpolation between personal + global best
                theta = np.random.random(self.dims)
                attractor = theta * p.best_pos + (1.0 - theta) * self.global_best_pos

                # Quantum displacement: beta * |mbest - x_i| * ln(1/u)
                displacement = beta * np.abs(mbest - p.position) * np.log(1.0 / u)

                # Quantum step (position from |ψ|² distribution)
                x_quantum = attractor + sign * displacement

                # Wave-packet momentum (de Broglie component p = ℏk)
                # A light directional nudge that helps particles follow
                # curved valleys without overwhelming quantum tunnelling
                # on multimodal landscapes.
                delta = x_quantum - prev_pos
                p.momentum = self.mom_decay * p.momentum + (1.0 - self.mom_decay) * delta

                # Final position: quantum step + momentum nudge
                p.position = np.clip(x_quantum + self.mom_weight * p.momentum, lo, hi)

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
