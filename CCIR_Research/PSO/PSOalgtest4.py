"""
Particle Swarm Optimization (PSO)
==================================
Each particle represents a potential solution in the search space.
Particles adjust their positions based on:
  - Their own best known position (personal best)
  - The swarm's global best known position (global best)

PSO Parameters:
  - num_particles  : number of particles in the swarm
  - dimensions     : number of dimensions in the search space
  - inertia_weight : controls how much of the previous velocity is retained (w)
  - cognitive_coef : weight toward the particle's personal best (c1)
  - social_coef    : weight toward the swarm's global best (c2)
  - max_iterations : stopping criterion
  - bounds         : (min, max) for each dimension
"""

import numpy as np
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════════╗
# ║                     USER SETTINGS                           ║
# ║  Change these values to control the PSO goal and behavior   ║
# ╚══════════════════════════════════════════════════════════════╝

#    Goal position — the target coordinates the swarm will search for.
#    Add or remove numbers to change the number of dimensions.
#    Example: [0, 0] is 2D,  [3, -5, 2] is 3D,  [1, 2, 3, 4] is 4D
GOAL = [3.0, -5.0, 2.0, -8.0]

#    Search space — particles will only search between these two values.
#    Make sure your GOAL coordinates fall within this range!
BOUNDS = (-10.0, 10.0)

#    Swarm settings
NUM_PARTICLES  = 40     # How many particles in the swarm
MAX_ITERATIONS = 200    # How many rounds the swarm runs for

#    PSO tuning parameters
INERTIA_WEIGHT = 0.729  # How much old velocity carries over (0-1). Higher = more exploration
COGNITIVE_COEF = 1.494  # Pull toward each particle's personal best
SOCIAL_COEF    = 1.494  # Pull toward the swarm's global best
INERTIA_DECAY  = 0.99   # Multiply inertia by this each iteration (1.0 = no decay)
VELOCITY_CLAMP = 0.2    # Max speed as a fraction of the search range

# ╚══════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────
# Objective / Fitness Functions
# ─────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    """Simple Sphere function — global minimum at origin (f=0)."""
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function — highly multimodal, global minimum at origin (f=0)."""
    A = 10
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock (banana) function — global minimum at (1,1,...,1) (f=0)."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


# ─────────────────────────────────────────────
# Particle Class
# ─────────────────────────────────────────────

class Particle:
    """
    Represents a single particle in the swarm.

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
        # Random initial position within bounds
        self.position: np.ndarray = np.random.uniform(lo, hi, dimensions)
        # Random initial velocity (fraction of the search range)
        velocity_range = (hi - lo) * 0.1
        self.velocity: np.ndarray = np.random.uniform(-velocity_range, velocity_range, dimensions)

        self.best_pos: np.ndarray = self.position.copy()
        self.best_score: float = float("inf")
        self.score: float = float("inf")

    def evaluate(self, objective_fn) -> float:
        """Evaluate the particle's current position and update personal best."""
        self.score = objective_fn(self.position)
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_pos = self.position.copy()
        return self.score

    def update_velocity(
        self,
        global_best_pos: np.ndarray,
        inertia_weight: float,
        cognitive_coef: float,
        social_coef: float,
    ) -> None:
        """
        Velocity update equation:
          v(t+1) = w * v(t)
                 + c1 * r1 * (personal_best - position)
                 + c2 * r2 * (global_best  - position)
        """
        r1 = np.random.random(len(self.position))
        r2 = np.random.random(len(self.position))

        cognitive = cognitive_coef * r1 * (self.best_pos - self.position)
        social    = social_coef    * r2 * (global_best_pos - self.position)

        self.velocity = inertia_weight * self.velocity + cognitive + social

    def update_position(self, bounds: tuple[float, float]) -> None:
        """Move the particle and clamp to search bounds."""
        self.position += self.velocity
        lo, hi = bounds
        self.position = np.clip(self.position, lo, hi)


# ─────────────────────────────────────────────
# PSO Engine
# ─────────────────────────────────────────────

class PSO:
    """
    Particle Swarm Optimizer.

    Parameters
    ----------
    objective_fn    : callable — function to minimise
    dimensions      : int — number of decision variables
    bounds          : (float, float) — search space limits for every dimension
    num_particles   : int — swarm size (default 30)
    max_iterations  : int — stopping criterion (default 200)
    inertia_weight  : float — w, balances exploration vs exploitation (default 0.7)
    cognitive_coef  : float — c1, personal-best attraction (default 1.5)
    social_coef     : float — c2, global-best attraction (default 1.5)
    inertia_decay   : float — multiply w by this factor each iteration (default 1.0)
    velocity_clamp  : float or None — max speed as fraction of search range
    """

    def __init__(
        self,
        objective_fn,
        dimensions: int     = 2,
        bounds: tuple       = (-5.12, 5.12),
        num_particles: int  = 30,
        max_iterations: int = 200,
        inertia_weight: float = 0.729,
        cognitive_coef: float = 1.494,
        social_coef: float    = 1.494,
        inertia_decay: float  = 1.0,
        velocity_clamp: float | None = 0.2,
    ):
        self.objective_fn   = objective_fn
        self.dimensions     = dimensions
        self.bounds         = bounds
        self.num_particles  = num_particles
        self.max_iterations = max_iterations
        self.w              = inertia_weight
        self.c1             = cognitive_coef
        self.c2             = social_coef
        self.inertia_decay  = inertia_decay
        self.v_max          = velocity_clamp * (bounds[1] - bounds[0]) if velocity_clamp else None

        # Swarm state
        self.swarm: list[Particle] = []
        self.global_best_pos: np.ndarray | None = None
        self.global_best_score: float = float("inf")

        # History for plotting
        self.best_score_history: list[float] = []
        self.avg_score_history:  list[float] = []

    def _init_swarm(self) -> None:
        self.swarm = [Particle(self.dimensions, self.bounds) for _ in range(self.num_particles)]

    def _enforce_velocity_clamp(self, particle: Particle) -> None:
        if self.v_max is not None:
            particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)

    def optimize(self, verbose: bool = True) -> tuple[np.ndarray, float]:
        """
        Run the PSO loop.

        Returns
        -------
        global_best_pos   : best position found
        global_best_score : fitness at that position
        """
        self._init_swarm()

        for iteration in range(1, self.max_iterations + 1):
            # ── Evaluate each particle ──────────────────────────
            scores = []
            for particle in self.swarm:
                score = particle.evaluate(self.objective_fn)
                scores.append(score)

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos   = particle.position.copy()

            # ── Record history ──────────────────────────────────
            self.best_score_history.append(self.global_best_score)
            self.avg_score_history.append(float(np.mean(scores)))

            # ── Update velocities & positions ───────────────────
            for particle in self.swarm:
                particle.update_velocity(
                    self.global_best_pos, self.w, self.c1, self.c2
                )
                self._enforce_velocity_clamp(particle)
                particle.update_position(self.bounds)

            # ── Inertia decay ────────────────────────────────────
            self.w *= self.inertia_decay

            # ── Logging ──────────────────────────────────────────
            if verbose and (iteration % 20 == 0 or iteration == 1):
                print(
                    f"Iter {iteration:>4d}/{self.max_iterations} | "
                    f"Best Score: {self.global_best_score:.6f} | "
                    f"Best Pos: {np.round(self.global_best_pos, 4)}"
                )

        return self.global_best_pos, self.global_best_score

    def plot_convergence(self, title: str = "PSO Convergence") -> None:
        """Plot best and average fitness over iterations."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_score_history, label="Global Best", linewidth=2, color="royalblue")
        plt.plot(self.avg_score_history,  label="Swarm Average", linewidth=1.5,
                 linestyle="--", color="tomato", alpha=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness (lower = better)")
        plt.title(title)
        plt.legend()
        plt.yscale("log")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("pso_convergence.png", dpi=150)
        plt.show()
        print("Convergence plot saved → pso_convergence.png")


# ─────────────────────────────────────────────
# Run PSO with User Settings
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # Build the objective function automatically from GOAL
    def custom_target(x: np.ndarray) -> float:
        """Shifted sphere — minimum is located exactly at GOAL coordinates."""
        goal = np.array(GOAL)
        return float(np.sum((x - goal) ** 2))

    dimensions = len(GOAL)

    print("=" * 60)
    print(f"PSO — Custom Target Function ({dimensions} dimensions)")
    print(f"🎯 Goal coordinates : {GOAL}")
    print(f"🔍 Search bounds    : {BOUNDS}")
    print(f"🐦 Particles        : {NUM_PARTICLES}")
    print(f"🔁 Iterations       : {MAX_ITERATIONS}")
    print("=" * 60)

    pso = PSO(
        objective_fn   = custom_target,
        dimensions     = dimensions,
        bounds         = BOUNDS,
        num_particles  = NUM_PARTICLES,
        max_iterations = MAX_ITERATIONS,
        inertia_weight = INERTIA_WEIGHT,
        cognitive_coef = COGNITIVE_COEF,
        social_coef    = SOCIAL_COEF,
        inertia_decay  = INERTIA_DECAY,
        velocity_clamp = VELOCITY_CLAMP,
    )

    best_pos, best_score = pso.optimize()

    print(f"\n✓ Best position found : {np.round(best_pos, 6)}")
    print(f"✓ Target goal         : {GOAL}")
    print(f"✓ Best score (0 = perfect) : {best_score:.10f}")
    pso.plot_convergence(f"PSO — Targeting {GOAL}")

    # ── Rastrigin (multimodal, 2-D) ──────────────────────────────
    print("\n" + "=" * 60)
    print("Rastrigin Function (2 dimensions, multimodal)")
    print("Expected global minimum: f(0, 0) = 0")
    print("=" * 60)

    pso_ras = PSO(
        objective_fn   = rastrigin,
        dimensions     = 2,
        bounds         = (-5.12, 5.12),
        num_particles  = 50,
        max_iterations = 300,
        inertia_weight = 0.9,
        cognitive_coef = 2.0,
        social_coef    = 2.0,
        inertia_decay  = 0.995,
        velocity_clamp = 0.3,
    )
    best_pos, best_score = pso_ras.optimize()
    print(f"\n✓ Best position found : {np.round(best_pos, 6)}")
    print(f"✓ Best score (0 = perfect) : {best_score:.10f}")
    pso_ras.plot_convergence("PSO — Rastrigin Function")

    # ── Rosenbrock (banana, 3-D) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Rosenbrock Function (3 dimensions)")
    print("Expected global minimum: f(1, 1, 1) = 0")
    print("=" * 60)

    pso_ros = PSO(
        objective_fn   = rosenbrock,
        dimensions     = 3,
        bounds         = (-2.0, 2.0),
        num_particles  = 60,
        max_iterations = 500,
        inertia_weight = 0.7,
        cognitive_coef = 1.8,
        social_coef    = 1.8,
        inertia_decay  = 0.998,
        velocity_clamp = 0.15,
    )
    best_pos, best_score = pso_ros.optimize()
    print(f"\n✓ Best position found : {np.round(best_pos, 6)}")
    print(f"✓ Best score (0 = perfect) : {best_score:.10f}")
    pso_ros.plot_convergence("PSO — Rosenbrock Function")
