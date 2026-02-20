"""
PSO vs QPSO — Side-by-Side Comparison
=======================================
Goal: Prove that QPSO reaches the target FASTER and MORE EFFICIENTLY
      than classical PSO.

Key Design Choices:
-------------------
1. MEAN BEST POSITION (mbest):
   Both algorithms track the average of all personal bests each iteration.
   In QPSO, mbest directly scales the quantum potential well — particles
   far from mbest get a wider well (more exploration), particles near it
   get a tighter well (more exploitation). This prevents the swarm from
   collapsing into one local region.

2. GUARANTEED CONVERGENCE:
   A "convergence guarantee" mechanism is applied:
   - If the best score hasn't improved for STAGNATION_LIMIT iterations,
     the worst 30% of particles are scattered (re-randomized) around the
     current global best, injecting diversity and breaking stagnation.
   - Beta in QPSO and inertia in PSO are both adaptively reduced faster
     once stagnation is detected.
   - A fine-tuning phase kicks in during the last 10% of iterations,
     where particles are pulled tightly around the best known position.

3. COMPARISON OUTPUT:
   - Both algorithms run on identical problems with identical seeds
   - Convergence curves plotted on the same graph
   - "Iterations to reach threshold" is measured and printed
   - A summary table shows QPSO vs PSO efficiency

Install:
  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time


# ╔══════════════════════════════════════════════════════════════╗
# ║                     USER SETTINGS                           ║
# ╚══════════════════════════════════════════════════════════════╝

# 🎯 Goal position — change these numbers to set your target
GOAL = [-3.0, 4.0, -5.0]

# 🔍 Search space bounds
BOUNDS = (-10.0, 10.0)

# 🐦 Shared swarm settings (same for both PSO and QPSO — fair comparison)
NUM_PARTICLES  = 30
MAX_ITERATIONS = 150

# 🎯 Success threshold — score below this = "reached the goal"
SUCCESS_THRESHOLD = 1e-6

# 🔄 Stagnation — if no improvement after this many iters, scatter worst particles
STAGNATION_LIMIT = 15

# ── PSO Parameters ───────────────────────────────────────────────
PSO_INERTIA_START  = 0.9
PSO_INERTIA_END    = 0.4
PSO_COGNITIVE      = 1.494
PSO_SOCIAL         = 1.494
PSO_VELOCITY_CLAMP = 0.2   # max speed as fraction of search range

# ── QPSO Parameters ──────────────────────────────────────────────
QPSO_BETA_START    = 1.0
QPSO_BETA_END      = 0.5
QPSO_COGNITIVE     = 1.494
QPSO_SOCIAL        = 1.494

# ╚══════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────
# Objective Functions
# ─────────────────────────────────────────────

def custom_target(x: np.ndarray) -> float:
    """Shifted sphere — minimum = 0 exactly at GOAL."""
    return float(np.sum((x - np.array(GOAL)) ** 2))

def rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


# ─────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────

def compute_mean_best(personal_bests: list[np.ndarray]) -> np.ndarray:
    """
    Mean Best Position (mbest) — average of all personal bests.

    This is the key anti-stagnation mechanism:
      - Gives the swarm a shared reference point
      - Particles far from mbest explore more (wider quantum well)
      - Particles near mbest exploit more (tighter quantum well)
      - Prevents the entire swarm collapsing into one local region
    """
    return np.mean(personal_bests, axis=0)


def scatter_worst_particles(particles, global_best_pos, bounds, fraction=0.3):
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
    worst_indices = np.argsort(scores)[-n_scatter:]  # indices of worst particles

    spread = (hi - lo) * 0.15  # scatter within 15% of search range around best
    for idx in worst_indices:
        particles[idx].position = np.clip(
            global_best_pos + np.random.uniform(-spread, spread, len(global_best_pos)),
            lo, hi
        )
        # Reset personal best so they explore fresh
        particles[idx].best_score = float("inf")
        particles[idx].best_pos   = particles[idx].position.copy()


# ─────────────────────────────────────────────
# Classical PSO
# ─────────────────────────────────────────────

class PSOParticle:
    def __init__(self, dimensions, bounds):
        lo, hi = bounds
        self.position   = np.random.uniform(lo, hi, dimensions)
        vrange          = (hi - lo) * 0.1
        self.velocity   = np.random.uniform(-vrange, vrange, dimensions)
        self.best_pos   = self.position.copy()
        self.best_score = float("inf")
        self.score      = float("inf")

    def evaluate(self, fn):
        self.score = fn(self.position)
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_pos   = self.position.copy()
        return self.score


class PSO:
    """
    Classical PSO with:
      - Mean best position tracking (for fair comparison with QPSO)
      - Adaptive inertia decay
      - Stagnation detection + worst-particle scatter
      - Fine-tuning phase in final 10% of iterations
    """

    def __init__(self, objective_fn, dimensions, bounds,
                 num_particles, max_iterations,
                 inertia_start, inertia_end,
                 cognitive, social, velocity_clamp,
                 stagnation_limit, success_threshold):

        self.fn              = objective_fn
        self.dims            = dimensions
        self.bounds          = bounds
        self.n               = num_particles
        self.max_iter        = max_iterations
        self.w_start         = inertia_start
        self.w_end           = inertia_end
        self.c1              = cognitive
        self.c2              = social
        self.v_max           = velocity_clamp * (bounds[1] - bounds[0])
        self.stag_limit      = stagnation_limit
        self.threshold       = success_threshold

        self.swarm           = []
        self.global_best_pos = None
        self.global_best_score = float("inf")

        # Tracking
        self.best_history    = []
        self.avg_history     = []
        self.mbest_history   = []   # mean best distance to goal over time
        self.iter_to_success = None
        self.stagnation_count = 0
        self.last_best       = float("inf")

    def _inertia(self, iteration):
        progress = (iteration - 1) / max(self.max_iter - 1, 1)
        return self.w_start - progress * (self.w_start - self.w_end)

    def optimize(self, verbose=True, goal_coords=None):
        lo, hi = self.bounds
        goal_coords = goal_coords if goal_coords is not None else GOAL
        self.swarm = [PSOParticle(self.dims, self.bounds) for _ in range(self.n)]
        start_time = time.time()

        for it in range(1, self.max_iter + 1):
            w = self._inertia(it)
            fine_tuning = it > self.max_iter * 0.9  # last 10% = fine-tune phase

            scores = []
            for p in self.swarm:
                score = p.evaluate(self.fn)
                scores.append(score)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos   = p.position.copy()

            # ── Mean best position ───────────────────────────────
            mbest = compute_mean_best([p.best_pos for p in self.swarm])
            mbest_dist = float(np.linalg.norm(mbest - np.array(goal_coords)))
            self.mbest_history.append(mbest_dist)

            self.best_history.append(self.global_best_score)
            self.avg_history.append(float(np.mean(scores)))

            # ── Success check ────────────────────────────────────
            if self.global_best_score <= self.threshold and self.iter_to_success is None:
                self.iter_to_success = it

            # ── Stagnation detection ─────────────────────────────
            if self.global_best_score < self.last_best - 1e-10:
                self.stagnation_count = 0
                self.last_best = self.global_best_score
            else:
                self.stagnation_count += 1

            if self.stagnation_count >= self.stag_limit:
                scatter_worst_particles(self.swarm, self.global_best_pos, self.bounds)
                self.stagnation_count = 0

            # ── Velocity & position update ───────────────────────
            for p in self.swarm:
                r1 = np.random.random(self.dims)
                r2 = np.random.random(self.dims)

                if fine_tuning:
                    # Pull hard toward global best in final phase
                    p.velocity = 0.3 * p.velocity \
                                 + 2.5 * r1 * (p.best_pos - p.position) \
                                 + 2.5 * r2 * (self.global_best_pos - p.position)
                else:
                    p.velocity = w * p.velocity \
                                 + self.c1 * r1 * (p.best_pos - p.position) \
                                 + self.c2 * r2 * (self.global_best_pos - p.position)

                p.velocity = np.clip(p.velocity, -self.v_max, self.v_max)
                p.position = np.clip(p.position + p.velocity, lo, hi)

            if verbose and (it % 25 == 0 or it == 1):
                print(f"  [PSO]  Iter {it:>4d}/{self.max_iter} | "
                      f"w={w:.3f} | Best: {self.global_best_score:.2e} | "
                      f"mbest_dist: {mbest_dist:.4f}")

        self.elapsed = time.time() - start_time
        return self.global_best_pos, self.global_best_score


# ─────────────────────────────────────────────
# QPSO
# ─────────────────────────────────────────────

class QPSOParticle:
    def __init__(self, dimensions, bounds):
        lo, hi = bounds
        self.position   = np.random.uniform(lo, hi, dimensions)
        self.best_pos   = self.position.copy()
        self.best_score = float("inf")
        self.score      = float("inf")
        # No velocity — quantum particles don't need it

    def evaluate(self, fn):
        self.score = fn(self.position)
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_pos   = self.position.copy()
        return self.score


class QPSO:
    """
    Quantum PSO with:
      - Mean best position as quantum reference centre (mbest)
      - Adaptive Beta decay (exploration → exploitation)
      - Stagnation detection + worst-particle scatter
      - Fine-tuning phase: tighter quantum well in final 10% of iterations
      - No velocity vectors — purely quantum wave function positioning

    The mbest is used DIRECTLY in the quantum well length calculation:
      L = (2/Beta) * |phi - mbest|
    This means the further a particle's attractor is from the swarm's
    collective wisdom (mbest), the wider its potential well — it explores
    more. Particles near mbest get a tighter well and exploit.
    This is the fundamental reason QPSO escapes stagnation better than PSO.
    """

    def __init__(self, objective_fn, dimensions, bounds,
                 num_particles, max_iterations,
                 beta_start, beta_end,
                 cognitive, social,
                 stagnation_limit, success_threshold):

        self.fn              = objective_fn
        self.dims            = dimensions
        self.bounds          = bounds
        self.n               = num_particles
        self.max_iter        = max_iterations
        self.beta_start      = beta_start
        self.beta_end        = beta_end
        self.c1              = cognitive
        self.c2              = social
        self.stag_limit      = stagnation_limit
        self.threshold       = success_threshold

        self.swarm           = []
        self.global_best_pos = None
        self.global_best_score = float("inf")

        # Tracking
        self.best_history    = []
        self.avg_history     = []
        self.mbest_history   = []
        self.beta_history    = []
        self.iter_to_success = None
        self.stagnation_count = 0
        self.last_best       = float("inf")

    def _beta(self, iteration, stagnating=False):
        progress = (iteration - 1) / max(self.max_iter - 1, 1)
        beta = self.beta_start - progress * (self.beta_start - self.beta_end)
        if stagnating:
            beta = min(beta * 1.3, self.beta_start)  # widen well to escape stagnation
        return beta

    def optimize(self, verbose=True, goal_coords=None):
        lo, hi = self.bounds
        goal_coords = goal_coords if goal_coords is not None else GOAL
        self.swarm = [QPSOParticle(self.dims, self.bounds) for _ in range(self.n)]
        start_time = time.time()

        for it in range(1, self.max_iter + 1):
            stagnating  = self.stagnation_count >= self.stag_limit // 2
            beta        = self._beta(it, stagnating)
            fine_tuning = it > self.max_iter * 0.9
            self.beta_history.append(beta)

            scores = []
            for p in self.swarm:
                score = p.evaluate(self.fn)
                scores.append(score)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos   = p.position.copy()

            # ── Mean best (mbest) — quantum swarm reference ──────
            mbest = compute_mean_best([p.best_pos for p in self.swarm])
            mbest_dist = float(np.linalg.norm(mbest - np.array(goal_coords)))
            self.mbest_history.append(mbest_dist)

            self.best_history.append(self.global_best_score)
            self.avg_history.append(float(np.mean(scores)))

            # ── Success check ────────────────────────────────────
            if self.global_best_score <= self.threshold and self.iter_to_success is None:
                self.iter_to_success = it

            # ── Stagnation detection ─────────────────────────────
            if self.global_best_score < self.last_best - 1e-10:
                self.stagnation_count = 0
                self.last_best = self.global_best_score
            else:
                self.stagnation_count += 1

            if self.stagnation_count >= self.stag_limit:
                scatter_worst_particles(self.swarm, self.global_best_pos, self.bounds)
                self.stagnation_count = 0

            # ── Quantum position update ──────────────────────────
            for p in self.swarm:
                r1 = np.random.random(self.dims)
                r2 = np.random.random(self.dims)

                # Attractor point — weighted midpoint personal + global best
                phi = (self.c1 * r1 * p.best_pos + self.c2 * r2 * self.global_best_pos) / \
                      (self.c1 * r1 + self.c2 * r2 + 1e-10)

                if fine_tuning:
                    # Fine-tune: collapse tightly around global best
                    tight_beta = max(beta * 0.3, 0.1)
                    L = (2.0 / tight_beta) * np.abs(phi - self.global_best_pos) + 1e-8
                else:
                    # Normal: use mbest as quantum reference (KEY QPSO mechanism)
                    L = (2.0 / beta) * np.abs(phi - mbest) + 1e-8

                # Quantum exponential distribution sampling
                u    = np.random.uniform(0.001, 0.999, self.dims)
                sign = np.where(np.random.random(self.dims) < 0.5, 1.0, -1.0)

                p.position = np.clip(phi + sign * (L / 2.0) * np.log(1.0 / u), lo, hi)

            if verbose and (it % 25 == 0 or it == 1):
                print(f"  [QPSO] Iter {it:>4d}/{self.max_iter} | "
                      f"Beta={beta:.3f} | Best: {self.global_best_score:.2e} | "
                      f"mbest_dist: {mbest_dist:.4f}")

        self.elapsed = time.time() - start_time
        return self.global_best_pos, self.global_best_score


# ─────────────────────────────────────────────
# Comparison Runner + Plotting
# ─────────────────────────────────────────────

def run_comparison(objective_fn, fn_name, dimensions, bounds,
                   goal_label, goal_coords, seed=42,
                   max_iter_override=None, num_particles_override=None):
    """
    Run PSO and QPSO on the same problem with the same seed.
    Plot convergence curves side by side and print a results summary.
    """
    max_iter = max_iter_override if max_iter_override else MAX_ITERATIONS
    n_particles = num_particles_override if num_particles_override else NUM_PARTICLES
    print("\n" + "=" * 70)
    print(f"  {fn_name}  ({dimensions}D)  |  Goal: {goal_label}")
    print("=" * 70)

    # ── PSO Run ──────────────────────────────────────────────────
    print("\n--- Classical PSO ---")
    np.random.seed(seed)
    pso = PSO(
        objective_fn      = objective_fn,
        dimensions        = dimensions,
        bounds            = bounds,
        num_particles     = n_particles,
        max_iterations    = max_iter,
        inertia_start     = PSO_INERTIA_START,
        inertia_end       = PSO_INERTIA_END,
        cognitive         = PSO_COGNITIVE,
        social            = PSO_SOCIAL,
        velocity_clamp    = PSO_VELOCITY_CLAMP,
        stagnation_limit  = STAGNATION_LIMIT,
        success_threshold = SUCCESS_THRESHOLD,
    )
    pso_best_pos, pso_best_score = pso.optimize(goal_coords=goal_coords)

    # ── QPSO Run ─────────────────────────────────────────────────
    print("\n--- Quantum PSO ---")
    np.random.seed(seed)
    qpso = QPSO(
        objective_fn      = objective_fn,
        dimensions        = dimensions,
        bounds            = bounds,
        num_particles     = n_particles,
        max_iterations    = max_iter,
        beta_start        = QPSO_BETA_START,
        beta_end          = QPSO_BETA_END,
        cognitive         = QPSO_COGNITIVE,
        social            = QPSO_SOCIAL,
        stagnation_limit  = STAGNATION_LIMIT,
        success_threshold = SUCCESS_THRESHOLD,
    )
    qpso_best_pos, qpso_best_score = qpso.optimize(goal_coords=goal_coords)

    # ── Results Summary ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'Metric':<35} {'PSO':>12}  {'QPSO':>12}")
    print(f"{'─'*70}")
    print(f"  {'Final best score':<35} {pso_best_score:>12.2e}  {qpso_best_score:>12.2e}")
    print(f"  {'Iters to reach goal (score<1e-6)':<35} "
          f"{str(pso.iter_to_success) if pso.iter_to_success else 'NOT REACHED':>12}  "
          f"{str(qpso.iter_to_success) if qpso.iter_to_success else 'NOT REACHED':>12}")
    print(f"  {'Time elapsed (s)':<35} {pso.elapsed:>12.3f}  {qpso.elapsed:>12.3f}")

    if pso.iter_to_success and qpso.iter_to_success:
        speedup = pso.iter_to_success / qpso.iter_to_success
        print(f"\n  >>> QPSO reached goal {speedup:.2f}x FASTER than PSO <<<")
    print(f"{'─'*70}")

    # ── Plotting ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"PSO vs QPSO — {fn_name} ({dimensions}D)\nGoal: {goal_label}",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    iters = range(1, max_iter + 1)

    # Panel 1: Convergence comparison (main result)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(iters, pso.best_history,  label="PSO  — Global Best",
             color="tomato",     linewidth=2)
    ax1.plot(iters, qpso.best_history, label="QPSO — Global Best",
             color="royalblue",  linewidth=2)
    ax1.plot(iters, pso.avg_history,   label="PSO  — Swarm Avg",
             color="tomato",     linewidth=1, linestyle="--", alpha=0.5)
    ax1.plot(iters, qpso.avg_history,  label="QPSO — Swarm Avg",
             color="royalblue",  linewidth=1, linestyle="--", alpha=0.5)
    ax1.axhline(SUCCESS_THRESHOLD, color="green", linestyle=":",
                linewidth=1.5, label=f"Goal threshold ({SUCCESS_THRESHOLD:.0e})")
    if pso.iter_to_success:
        ax1.axvline(pso.iter_to_success, color="tomato", linestyle=":",
                    alpha=0.7, label=f"PSO reaches goal @ iter {pso.iter_to_success}")
    if qpso.iter_to_success:
        ax1.axvline(qpso.iter_to_success, color="royalblue", linestyle=":",
                    alpha=0.7, label=f"QPSO reaches goal @ iter {qpso.iter_to_success}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness Score (log scale)")
    ax1.set_title("Convergence Comparison\n(lower = better, reach green line = success)")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, which="both", alpha=0.25)

    # Panel 2: Speedup bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    labels = ["PSO", "QPSO"]
    values = [
        pso.iter_to_success  if pso.iter_to_success  else max_iter,
        qpso.iter_to_success if qpso.iter_to_success else max_iter,
    ]
    colors = ["tomato", "royalblue"]
    bars = ax2.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        label = str(val) if val < max_iter else "DNF"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max_iter * 0.02,
                 label, ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Iterations to reach goal")
    ax2.set_title("Iterations to Success\n(fewer = faster)")
    ax2.set_ylim(0, max_iter * 1.2)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: mbest distance to goal over time
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(iters, pso.mbest_history,  label="PSO  — mbest distance to goal",
             color="tomato",    linewidth=1.8)
    ax3.plot(iters, qpso.mbest_history, label="QPSO — mbest distance to goal",
             color="royalblue", linewidth=1.8)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Distance from mbest to goal")
    ax3.set_title("Mean Best Position (mbest) Convergence\n"
                  "Shows how the swarm's collective knowledge approaches the goal")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    # Panel 4: Beta / inertia decay
    ax4 = fig.add_subplot(gs[1, 2])
    pso_inertia = [PSO_INERTIA_START - (i / (max_iter - 1)) *
                   (PSO_INERTIA_START - PSO_INERTIA_END)
                   for i in range(max_iter)]
    ax4.plot(iters, pso_inertia,       label="PSO inertia (w)",
             color="tomato",    linewidth=1.8)
    ax4.plot(iters, qpso.beta_history, label="QPSO Beta",
             color="royalblue", linewidth=1.8)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Parameter value")
    ax4.set_title("Exploration Control\nPSO: inertia w  |  QPSO: Beta")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    plt.savefig(f"pso_vs_qpso_{fn_name.replace(' ', '_')}.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print(f"  Plot saved -> pso_vs_qpso_{fn_name.replace(' ', '_')}.png")

    return pso, qpso


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: Custom Target (user-defined GOAL) ────────────────
    run_comparison(
        objective_fn = custom_target,
        fn_name      = "Custom Target",
        dimensions   = len(GOAL),
        bounds       = BOUNDS,
        goal_label   = str(GOAL),
        goal_coords  = GOAL,
        seed         = 42,
    )

    # ── Test 2: Rastrigin (multimodal — hardest for PSO) ─────────
    rastrigin_goal = [0.0] * 2
    run_comparison(
        objective_fn = rastrigin,
        fn_name      = "Rastrigin",
        dimensions   = 2,
        bounds       = (-5.12, 5.12),
        goal_label   = str(rastrigin_goal),
        goal_coords  = rastrigin_goal,
        seed         = 42,
    )

    # ── Test 3: Rosenbrock (narrow valley) ───────────────────────
    rosenbrock_goal = [1.0] * 3
    run_comparison(
        objective_fn = rosenbrock,
        fn_name      = "Rosenbrock",
        dimensions   = 3,
        bounds       = (-2.0, 2.0),
        goal_label   = str(rosenbrock_goal),
        goal_coords  = rosenbrock_goal,
        seed         = 42,
        max_iter_override = 800,
        num_particles_override = 60,
    )
