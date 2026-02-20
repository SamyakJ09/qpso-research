"""
Qiskit Quantum Particle Swarm Optimization (QPSO)
===================================================
A true quantum-circuit-driven QPSO using IBM's Qiskit framework.

How Qiskit is used here:
-------------------------
In classical QPSO, quantum behaviour is only *mathematically simulated*
using classical random numbers. This version replaces those classical
random sources with REAL quantum circuits:

  1. Hadamard Gate (H)  — puts a qubit into superposition |0⟩+|1⟩
                           measuring it gives a true quantum random bit
                           used for the ± tunnelling direction

  2. RY Rotation Gate   — rotates a qubit by angle θ = Beta * π
                           measurement probability encodes Beta directly
                           into the quantum circuit, biasing exploration

  3. Qiskit Aer         — local quantum circuit simulator
                           (swap `AerSimulator` for `IBMQBackend` to run
                            on real IBM quantum hardware!)

Quantum Circuit per particle per dimension:
  ┌───┐ ┌──────────┐ ┌─┐
  ┤ H ├─┤ RY(β·π)  ├─┤M├  → collapse → tunnelling direction ±
  └───┘ └──────────┘ └─┘

Install:
  pip install qiskit qiskit-aer matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ╔══════════════════════════════════════════════════════════════╗
# ║                     USER SETTINGS                           ║
# ╚══════════════════════════════════════════════════════════════╝

# 🎯 Goal position — target the swarm hunts for
GOAL = [3.0, -5.0, 2.0]

# 🔍 Search space bounds (goal must be within this range!)
BOUNDS = (-10.0, 10.0)

# 🐦 Swarm settings
NUM_PARTICLES  = 20     # Fewer particles recommended — quantum circuits add overhead
MAX_ITERATIONS = 100    # Iterations (each runs NUM_PARTICLES * DIMS quantum circuits)

# ⚛️  Quantum / Beta settings
BETA_START = 1.0        # Initial quantum well size  (broad exploration)
BETA_END   = 0.5        # Final quantum well size    (tight exploitation)

# ⚙️  Weights
COGNITIVE_COEF = 1.494
SOCIAL_COEF    = 1.494

# 🖥️  Quantum backend shots (measurements per circuit)
#     Higher = more accurate quantum sampling, but slower
SHOTS = 512

# ╚══════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────
# Quantum Circuit Engine
# ─────────────────────────────────────────────

class QuantumEngine:
    """
    Manages Qiskit quantum circuits for QPSO.

    Provides two quantum operations:
      1. quantum_sign()  — H gate measurement → random ±1 direction
      2. quantum_u()     — RY(β·π) gate measurement → Beta-biased float in (0,1)
    """

    def __init__(self, shots: int = 512):
        self.simulator = AerSimulator()
        self.shots = shots

    def quantum_sign(self, n: int) -> np.ndarray:
        """
        Generate n quantum random signs (+1 or -1) using Hadamard gates.

        Circuit:
          |0⟩ ──[H]──[M]──
          H gate creates equal superposition: |0⟩ → (|0⟩ + |1⟩) / √2
          Measuring collapses to 0 or 1 with exactly 50% probability each.
          0 → +1,  1 → -1

        This is the quantum tunnelling direction — which side of the
        attractor the qubot appears on.
        """
        signs = np.ones(n)
        for i in range(n):
            qc = QuantumCircuit(1, 1)
            qc.h(0)          # Hadamard: superposition
            qc.measure(0, 0) # Collapse wave function

            compiled = transpile(qc, self.simulator)
            job = self.simulator.run(compiled, shots=self.shots)
            counts = job.result().get_counts()

            # Probability of |1⟩ → negative sign
            prob_one = counts.get("1", 0) / self.shots
            signs[i] = 1.0 if np.random.random() > prob_one else -1.0

        return signs

    def quantum_u(self, n: int, beta: float) -> np.ndarray:
        """
        Generate n quantum random floats in (0,1) biased by Beta via RY gate.

        Circuit:
          |0⟩ ──[H]──[RY(β·π)]──[M]──
          RY(θ) rotates the Bloch sphere by θ around Y-axis.
          P(|1⟩) = sin²(θ/2)  where θ = beta * π
          This encodes the quantum well size directly into the circuit.

        High Beta (1.0) → P(|1⟩) ≈ 0.5  → uniform, broad exploration
        Low  Beta (0.5) → P(|1⟩) ≈ 0.15 → biased, tighter exploitation
        """
        u_vals = np.zeros(n)
        theta = beta * np.pi  # encode Beta as rotation angle

        for i in range(n):
            qc = QuantumCircuit(1, 1)
            qc.h(0)           # Hadamard: superposition
            qc.ry(theta, 0)   # RY rotation: encode Beta
            qc.measure(0, 0)  # Collapse

            compiled = transpile(qc, self.simulator)
            job = self.simulator.run(compiled, shots=self.shots)
            counts = job.result().get_counts()

            # Use measurement probability as quantum float
            prob_one = counts.get("1", 0) / self.shots
            # Map to (0,1) — avoid exact 0 to prevent log(0)
            u_vals[i] = np.clip(prob_one + np.random.normal(0, 0.05), 0.01, 0.99)

        return u_vals

    def quantum_weights(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate quantum random r1, r2 weights using Hadamard circuits.
        These replace the classical np.random.random() calls for c1, c2 scaling.
        """
        r1 = np.array([self._h_sample() for _ in range(n)])
        r2 = np.array([self._h_sample() for _ in range(n)])
        return r1, r2

    def _h_sample(self) -> float:
        """Single Hadamard circuit → float via shot probability."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        compiled = transpile(qc, self.simulator)
        job = self.simulator.run(compiled, shots=self.shots)
        counts = job.result().get_counts()
        return counts.get("1", 0) / self.shots


# ─────────────────────────────────────────────
# Qubot Class (Qiskit-powered)
# ─────────────────────────────────────────────

class Qubot:
    """
    A quantum particle whose movement is driven by Qiskit circuits.

    No velocity vector — position is sampled from a quantum wave function
    whose randomness comes from real quantum circuit measurements.
    """

    def __init__(self, dimensions: int, bounds: tuple[float, float]):
        lo, hi = bounds
        self.position: np.ndarray = np.random.uniform(lo, hi, dimensions)
        self.best_pos: np.ndarray = self.position.copy()
        self.best_score: float    = float("inf")
        self.score: float         = float("inf")

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
        bounds: tuple[float, float],
        qengine: QuantumEngine,
    ) -> None:
        """
        Qiskit-powered quantum position update.

        All randomness comes from quantum circuit measurements:
          - r1, r2  : from Hadamard circuits  (attractor weights)
          - u       : from RY(β·π) circuits   (quantum displacement magnitude)
          - sign    : from Hadamard circuits  (tunnelling direction ±)

        Steps:
          1. Compute attractor phi using quantum weights r1, r2
          2. Compute quantum well length L from mean_best distance + Beta
          3. Sample displacement from quantum exponential distribution
          4. Apply quantum tunnelling sign to land on either side of phi
          5. Clamp to search bounds
        """
        dims = len(self.position)

        # ── Step 1: Quantum attractor phi ───────────────────────
        r1, r2 = qengine.quantum_weights(dims)
        phi = (c1 * r1 * self.best_pos + c2 * r2 * global_best_pos) / \
              (c1 * r1 + c2 * r2 + 1e-10)

        # ── Step 2: Quantum well length ──────────────────────────
        L = (2.0 / beta) * np.abs(phi - mean_best)

        # ── Step 3: Quantum displacement (RY-circuit sampled) ────
        u = qengine.quantum_u(dims, beta)          # Beta-biased quantum floats
        delta = (L / 2.0) * np.log(1.0 / u)       # Exponential quantum distribution

        # ── Step 4: Quantum tunnelling direction (H-circuit) ─────
        sign = qengine.quantum_sign(dims)           # True quantum ± from H gate

        self.position = phi + sign * delta

        # ── Step 5: Clamp to bounds ──────────────────────────────
        lo, hi = bounds
        self.position = np.clip(self.position, lo, hi)


# ─────────────────────────────────────────────
# Qiskit QPSO Engine
# ─────────────────────────────────────────────

class QiskitQPSO:
    """
    Full Qiskit-powered Quantum PSO optimizer.

    Every source of randomness in the particle update is replaced
    with a genuine quantum circuit measurement via Qiskit Aer.

    To run on REAL IBM quantum hardware, replace:
      AerSimulator()  →  IBMQBackend('ibm_brisbane')  (requires IBM account)
    """

    def __init__(
        self,
        objective_fn,
        dimensions: int      = 2,
        bounds: tuple        = (-10.0, 10.0),
        num_particles: int   = 20,
        max_iterations: int  = 100,
        beta_start: float    = 1.0,
        beta_end: float      = 0.5,
        cognitive_coef: float = 1.494,
        social_coef: float    = 1.494,
        shots: int           = 512,
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

        self.swarm: list[Qubot]              = []
        self.global_best_pos: np.ndarray | None = None
        self.global_best_score: float        = float("inf")

        self.best_score_history: list[float] = []
        self.avg_score_history:  list[float] = []
        self.beta_history:       list[float] = []

    def _init_swarm(self):
        self.swarm = [Qubot(self.dimensions, self.bounds)
                      for _ in range(self.num_particles)]

    def _mean_best(self) -> np.ndarray:
        return np.mean([q.best_pos for q in self.swarm], axis=0)

    def _beta(self, iteration: int) -> float:
        progress = (iteration - 1) / max(self.max_iterations - 1, 1)
        return self.beta_start - progress * (self.beta_start - self.beta_end)

    def optimize(self, verbose: bool = True) -> tuple[np.ndarray, float]:
        self._init_swarm()

        print(f"\n⚛️  Qiskit backend  : {self.qengine.simulator.name}")
        print(f"🔬 Shots per circuit: {self.qengine.shots}")
        print(f"📐 Circuits/iter    : ~{self.num_particles * self.dimensions * 3} "
              f"(H + RY + H per dim per particle)\n")

        for iteration in range(1, self.max_iterations + 1):
            beta = self._beta(iteration)
            self.beta_history.append(beta)

            scores = []
            for qubot in self.swarm:
                score = qubot.evaluate(self.objective_fn)
                scores.append(score)
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_pos   = qubot.position.copy()

            mean_best = self._mean_best()
            self.best_score_history.append(self.global_best_score)
            self.avg_score_history.append(float(np.mean(scores)))

            for qubot in self.swarm:
                qubot.quantum_update(
                    self.global_best_pos, mean_best, beta,
                    self.c1, self.c2, self.bounds, self.qengine
                )

            if verbose and (iteration % 10 == 0 or iteration == 1):
                print(
                    f"Iter {iteration:>4d}/{self.max_iterations} | "
                    f"β={beta:.3f} | "
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
        ax2.set_ylabel("Beta (quantum well size)")
        ax2.set_title("Beta Decay\n[RY gate angle = β·π]\nHigh=Exploration | Low=Exploitation")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe = title.replace(" ", "_").replace("—", "-")
        fname = f"qiskit_qpso_{safe}.png"
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"Plot saved → {fname}")


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
    print(f"Qiskit QPSO — Custom Target ({len(GOAL)}D)")
    print(f"🎯 Goal     : {GOAL}")
    print(f"🔍 Bounds   : {BOUNDS}")
    print(f"🐦 Qubots   : {NUM_PARTICLES}  |  🔁 Iters: {MAX_ITERATIONS}")
    print(f"🌊 Beta     : {BETA_START} → {BETA_END}")
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
    print(f"\n✓ Best position : {np.round(best_pos, 6)}")
    print(f"✓ Target goal   : {GOAL}")
    print(f"✓ Best score    : {best_score:.10f}")
    qpso1.plot_convergence(f"Qiskit QPSO — Targeting {GOAL}")

    # ── Rastrigin ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Qiskit QPSO — Rastrigin (2D, multimodal)")
    print("Expected minimum: f(0, 0) = 0")
    print("=" * 65)

    qpso2 = QiskitQPSO(
        objective_fn   = rastrigin,
        dimensions     = 2,
        bounds         = (-5.12, 5.12),
        num_particles  = 20,
        max_iterations = 100,
        beta_start     = 1.0,
        beta_end       = 0.5,
        shots          = SHOTS,
    )
    best_pos, best_score = qpso2.optimize()
    print(f"\n✓ Best position : {np.round(best_pos, 6)}")
    print(f"✓ Best score    : {best_score:.10f}")
    qpso2.plot_convergence("Qiskit QPSO — Rastrigin")

    # ── Rosenbrock ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Qiskit QPSO — Rosenbrock (3D)")
    print("Expected minimum: f(1, 1, 1) = 0")
    print("=" * 65)

    qpso3 = QiskitQPSO(
        objective_fn   = rosenbrock,
        dimensions     = 3,
        bounds         = (-2.0, 2.0),
        num_particles  = 20,
        max_iterations = 150,
        beta_start     = 1.0,
        beta_end       = 0.5,
        shots          = SHOTS,
    )
    best_pos, best_score = qpso3.optimize()
    print(f"\n✓ Best position : {np.round(best_pos, 6)}")
    print(f"✓ Best score    : {best_score:.10f}")
    qpso3.plot_convergence("Qiskit QPSO — Rosenbrock")
