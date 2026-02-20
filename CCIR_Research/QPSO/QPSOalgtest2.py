"""
Qiskit Quantum Particle Swarm Optimization (QPSO)
===================================================
Fixed version — avoids broken transpile(qc, AerSimulator) call.
Uses backend.run(qc) directly, and batches ALL circuits per iteration
into a single job to fix hanging and massive slowdown.

Install:
  pip install qiskit qiskit-aer matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ╔══════════════════════════════════════════════════════════════╗
# ║                     USER SETTINGS                            ║
# ╚══════════════════════════════════════════════════════════════╝

# Goal position — target the swarm hunts for
GOAL = [3.0, -5.0, 2.0]

# Search space bounds (goal must be within this range!)
BOUNDS = (-10.0, 10.0)

# Swarm settings
NUM_PARTICLES  = 20
MAX_ITERATIONS = 100

# Quantum Beta settings
BETA_START = 1.0    # Broad exploration early
BETA_END   = 0.5    # Tight exploitation late

# Weights
COGNITIVE_COEF = 1.494
SOCIAL_COEF    = 1.494

# Shots per circuit (higher = more accurate, slower)
SHOTS = 256

# ╚══════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────
# Quantum Engine  (batched, no transpile)
# ─────────────────────────────────────────────

class QuantumEngine:
    """
    Runs ALL quantum circuits for a full iteration in ONE batched job.

    Key fix: uses backend.run(circuits) directly instead of
    transpile(qc, backend) which is broken in Qiskit >= 1.x + Aer >= 0.14.

    Per particle per dimension, 3 circuits are built:
      [0] H gate            → quantum weight r1
      [1] H gate            → quantum weight r2
      [2] H + RY(β·π) gate  → Beta-biased displacement u + sign
    """

    def __init__(self, shots: int = 256):
        self.simulator = AerSimulator()
        self.shots = shots

    def _h_circuit(self) -> QuantumCircuit:
        """Single qubit Hadamard + measure → uniform quantum random bit."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        return qc

    def _ry_circuit(self, beta: float) -> QuantumCircuit:
        """H + RY(β·π) + measure → Beta-biased quantum random bit."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.ry(beta * np.pi, 0)
        qc.measure(0, 0)
        return qc

    def _prob_one(self, counts: dict) -> float:
        """Convert shot counts to P(|1>) in (0.01, 0.99)."""
        p = counts.get("1", 0) / self.shots
        return float(np.clip(p, 0.01, 0.99))

    def sample_batch(
        self,
        n_particles: int,
        dims: int,
        beta: float,
    ) -> tuple:
        """
        Build and run ALL circuits for one full swarm iteration in one job.

        Returns four arrays each of shape (n_particles, dims):
          r1   : quantum weights for cognitive term
          r2   : quantum weights for social term
          u    : Beta-biased displacement magnitudes
          sign : quantum tunnelling directions (+1 or -1)
        """
        total = n_particles * dims

        # Build circuit list: for each (particle, dim) -> 3 circuits
        # Order: [r1_circuit, r2_circuit, ry_circuit] x (n_particles x dims)
        circuits = []
        for _ in range(total):
            circuits.append(self._h_circuit())       # r1
            circuits.append(self._h_circuit())       # r2
            circuits.append(self._ry_circuit(beta))  # u + sign

        # ── Single batched job — no transpile needed ─────────────
        job = self.simulator.run(circuits, shots=self.shots)
        results = job.result()

        # ── Unpack results ───────────────────────────────────────
        r1_flat   = np.zeros(total)
        r2_flat   = np.zeros(total)
        u_flat    = np.zeros(total)
        sign_flat = np.ones(total)

        for i in range(total):
            base = i * 3
            r1_flat[i]  = self._prob_one(results.get_counts(base))
            r2_flat[i]  = self._prob_one(results.get_counts(base + 1))
            ry_prob      = self._prob_one(results.get_counts(base + 2))
            u_flat[i]   = ry_prob
            # Quantum tunnelling sign — derived from RY circuit measurement
            sign_flat[i] = 1.0 if ry_prob >= 0.5 else -1.0

        # Reshape to (n_particles, dims)
        return (
            r1_flat.reshape(n_particles, dims),
            r2_flat.reshape(n_particles, dims),
            u_flat.reshape(n_particles, dims),
            sign_flat.reshape(n_particles, dims),
        )


# ─────────────────────────────────────────────
# Qubot
# ─────────────────────────────────────────────

class Qubot:
    """Quantum particle — no velocity, position driven by wave function."""

    def __init__(self, dimensions: int, bounds: tuple):
        lo, hi = bounds
        self.position   = np.random.uniform(lo, hi, dimensions)
        self.best_pos   = self.position.copy()
        self.best_score = float("inf")
        self.score      = float("inf")

    def evaluate(self, objective_fn) -> float:
        self.score = objective_fn(self.position)
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_pos   = self.position.copy()
        return self.score

    def quantum_update(
        self,
        global_best_pos: np.ndarray,
        mean_best: np.ndarray,
        beta: float,
        c1: float,
        c2: float,
        bounds: tuple,
        r1: np.ndarray,
        r2: np.ndarray,
        u: np.ndarray,
        sign: np.ndarray,
    ) -> None:
        """
        Quantum position update using pre-sampled quantum circuit results.

          phi   = attractor (weighted midpoint of personal + global best)
          L     = quantum well length (scaled by Beta + distance to mbest)
          delta = exponential quantum displacement
          sign  = tunnelling direction from RY circuit measurement
        """
        # Attractor point
        phi = (c1 * r1 * self.best_pos + c2 * r2 * global_best_pos) / \
              (c1 * r1 + c2 * r2 + 1e-10)

        # Quantum well length
        L = (2.0 / beta) * np.abs(phi - mean_best)

        # Exponential displacement from quantum u samples
        delta = (L / 2.0) * np.log(1.0 / u)

        # Apply quantum tunnelling
        self.position = phi + sign * delta

        # Clamp to bounds
        lo, hi = bounds
        self.position = np.clip(self.position, lo, hi)


# ─────────────────────────────────────────────
# Qiskit QPSO Engine
# ─────────────────────────────────────────────

class QiskitQPSO:
    """Full Qiskit-powered QPSO — all randomness from quantum circuits."""

    def __init__(
        self,
        objective_fn,
        dimensions: int       = 2,
        bounds: tuple         = (-10.0, 10.0),
        num_particles: int    = 20,
        max_iterations: int   = 100,
        beta_start: float     = 1.0,
        beta_end: float       = 0.5,
        cognitive_coef: float = 1.494,
        social_coef: float    = 1.494,
        shots: int            = 256,
    ):
        self.objective_fn   = objective_fn
        self.dimensions     = dimensions
        self.bounds         = bounds
        self.num_particles  = num_particles
        self.max_iterations = max_iterations
        self.beta_start     = beta_start
        self.beta_end       = beta_end
        self.c1             = cognitive_coef
        self.c2             = social_coef

        self.qengine = QuantumEngine(shots=shots)

        self.swarm              = []
        self.global_best_pos    = None
        self.global_best_score  = float("inf")
        self.best_score_history = []
        self.avg_score_history  = []
        self.beta_history       = []

    def _init_swarm(self):
        self.swarm = [Qubot(self.dimensions, self.bounds)
                      for _ in range(self.num_particles)]

    def _mean_best(self) -> np.ndarray:
        return np.mean([q.best_pos for q in self.swarm], axis=0)

    def _beta(self, iteration: int) -> float:
        progress = (iteration - 1) / max(self.max_iterations - 1, 1)
        return self.beta_start - progress * (self.beta_start - self.beta_end)

    def optimize(self, verbose: bool = True) -> tuple:
        self._init_swarm()

        circuits_per_iter = self.num_particles * self.dimensions * 3
        print(f"\n  Backend         : {self.qengine.simulator.name}")
        print(f"  Shots/circuit   : {self.qengine.shots}")
        print(f"  Circuits/iter   : {circuits_per_iter} (batched into 1 job)\n")

        for iteration in range(1, self.max_iterations + 1):
            beta = self._beta(iteration)
            self.beta_history.append(beta)

            # ── Evaluate all qubots ──────────────────────────────
            scores = []
            for qubot in self.swarm:
                score = qubot.evaluate(self.objective_fn)
                scores.append(score)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos   = qubot.position.copy()

            self.best_score_history.append(self.global_best_score)
            self.avg_score_history.append(float(np.mean(scores)))

            # ── Single batched quantum job for ALL particles ──────
            mean_best = self._mean_best()
            r1_all, r2_all, u_all, sign_all = self.qengine.sample_batch(
                self.num_particles, self.dimensions, beta
            )

            # ── Update each qubot with its quantum samples ────────
            for i, qubot in enumerate(self.swarm):
                qubot.quantum_update(
                    self.global_best_pos, mean_best, beta,
                    self.c1, self.c2, self.bounds,
                    r1_all[i], r2_all[i], u_all[i], sign_all[i],
                )

            if verbose and (iteration % 10 == 0 or iteration == 1):
                print(
                    f"Iter {iteration:>4d}/{self.max_iterations} | "
                    f"Beta={beta:.3f} | "
                    f"Best: {self.global_best_score:.8f} | "
                    f"Pos: {np.round(self.global_best_pos, 4)}"
                )

        return self.global_best_pos, self.global_best_score

    def plot_convergence(self, title: str = "Qiskit QPSO") -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        ax1.plot(self.best_score_history, label="Global Best",
                 linewidth=2, color="royalblue")
        ax1.plot(self.avg_score_history, label="Swarm Average",
                 linewidth=1.5, linestyle="--", color="tomato", alpha=0.8)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Fitness (lower = better)")
        ax1.set_title("Fitness Convergence")
        ax1.legend()
        ax1.set_yscale("log")
        ax1.grid(True, which="both", alpha=0.3)

        ax2.plot(self.beta_history, color="mediumpurple", linewidth=2)
        ax2.fill_between(range(len(self.beta_history)),
                         self.beta_history, alpha=0.15, color="mediumpurple")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Beta")
        ax2.set_title("Beta Decay — RY gate angle = Beta x pi\nHigh=Exploration | Low=Exploitation")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe = title.replace(" ", "_").replace(",", "").replace("[","").replace("]","")
        fname = f"qiskit_qpso_{safe}.png"
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"Plot saved -> {fname}")


# ─────────────────────────────────────────────
# Objective Functions
# ─────────────────────────────────────────────

def custom_target(x: np.ndarray) -> float:
    goal = np.array(GOAL)
    return float(np.sum((x - goal) ** 2))

def rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ── Custom Target ────────────────────────────────────────────
    print("=" * 65)
    print(f"Qiskit QPSO -- Custom Target ({len(GOAL)}D)")
    print(f"Goal   : {GOAL}")
    print(f"Bounds : {BOUNDS}")
    print(f"Qubots : {NUM_PARTICLES}  |  Iters: {MAX_ITERATIONS}")
    print(f"Beta   : {BETA_START} -> {BETA_END}")
    print("=" * 65)

    qpso1 = QiskitQPSO(
        objective_fn   = custom_target,
        dimensions     = len(GOAL),
        bounds         = BOUNDS,
        num_particles  = NUM_PARTICLES,
        max_iterations = MAX_ITERATIONS,
        beta_start     = BETA_START,
        beta_end       = BETA_END,
        cognitive_coef = COGNITIVE_COEF,
        social_coef    = SOCIAL_COEF,
        shots          = SHOTS,
    )
    best_pos, best_score = qpso1.optimize()
    print(f"\n  Best position : {np.round(best_pos, 6)}")
    print(f"  Target goal   : {GOAL}")
    print(f"  Best score    : {best_score:.10f}")
    qpso1.plot_convergence(f"Qiskit QPSO -- Targeting {GOAL}")

    # ── Rastrigin ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Qiskit QPSO -- Rastrigin (2D, multimodal)")
    print("Expected minimum: f(0, 0) = 0")
    print("=" * 65)

    qpso2 = QiskitQPSO(
        objective_fn   = rastrigin,
        dimensions     = 2,
        bounds         = (-5.12, 5.12),
        num_particles  = NUM_PARTICLES,
        max_iterations = MAX_ITERATIONS,
        beta_start     = 1.0,
        beta_end       = 0.5,
        shots          = SHOTS,
    )
    best_pos, best_score = qpso2.optimize()
    print(f"\n  Best position : {np.round(best_pos, 6)}")
    print(f"  Best score    : {best_score:.10f}")
    qpso2.plot_convergence("Qiskit QPSO -- Rastrigin")

    # ── Rosenbrock ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Qiskit QPSO -- Rosenbrock (3D)")
    print("Expected minimum: f(1, 1, 1) = 0")
    print("=" * 65)

    qpso3 = QiskitQPSO(
        objective_fn   = rosenbrock,
        dimensions     = 3,
        bounds         = (-2.0, 2.0),
        num_particles  = NUM_PARTICLES,
        max_iterations = MAX_ITERATIONS,
        beta_start     = 1.0,
        beta_end       = 0.5,
        shots          = SHOTS,
    )
    best_pos, best_score = qpso3.optimize()
    print(f"\n  Best position : {np.round(best_pos, 6)}")
    print(f"  Best score    : {best_score:.10f}")
    qpso3.plot_convergence("Qiskit QPSO -- Rosenbrock")
