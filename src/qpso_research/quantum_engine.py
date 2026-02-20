"""
Qiskit quantum circuit engine for QPSO.

Provides two sampling modes:
  - sample_batch_full(): ALL randomness (r1, r2, u, sign) from quantum circuits
  - sample_signs_only(): only tunnelling sign from quantum, rest classical

Uses backend.run(circuits) directly — avoids the broken transpile() call
that causes crashes in Qiskit >= 1.x + Aer >= 0.14. All circuits per
iteration are batched into a single job for performance.

Qiskit is imported lazily so that mode="math" works without it installed.
"""

import numpy as np


def _lazy_import_qiskit():
    """Import Qiskit components on first use."""
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        return QuantumCircuit, AerSimulator
    except ImportError as e:
        raise ImportError(
            "Qiskit is required for quantum modes (full/hybrid). "
            "Install with: pip install qiskit qiskit-aer"
        ) from e


class QuantumEngine:
    """
    Manages Qiskit quantum circuits for QPSO.

    Two modes of operation:
      1. sample_batch_full() — for mode="full": ALL random values from circuits
         Three circuit types per (particle x dimension):
           H gate            -> uniform quantum float (r1 weight)
           H gate            -> uniform quantum float (r2 weight)
           H + RY(beta*pi)   -> Beta-biased float (displacement u + sign)

      2. sample_signs_only() — for mode="hybrid": only tunnelling sign from circuits
         One RY circuit per (particle x dimension):
           H + RY(beta*pi)   -> Beta-biased sign (+1 or -1)
         r1, r2, u remain classical numpy for float precision.

    Why the split? Shot-based measurements only produce values in steps of
    1/SHOTS. With SHOTS=256, that's 0.004 granularity — too coarse for
    continuous math. Quantum gates belong in BINARY decisions where they
    provide genuine randomness.
    """

    def __init__(self, shots: int = 256):
        QuantumCircuit, AerSimulator = _lazy_import_qiskit()
        self._QuantumCircuit = QuantumCircuit
        self.backend = AerSimulator()
        self.shots = shots

    def _h_circuit(self):
        """Hadamard + measure -> true quantum random bit (P=0.5 each side)."""
        qc = self._QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        return qc

    def _ry_circuit(self, beta: float):
        """
        H + RY(beta*pi) + measure -> Beta-biased quantum random bit.
        P(|1>) = sin^2(beta*pi/2).
        High Beta -> P~0.5 (broad)  |  Low Beta -> P~0.15 (tight)
        """
        qc = self._QuantumCircuit(1, 1)
        qc.h(0)
        qc.ry(beta * np.pi, 0)
        qc.measure(0, 0)
        return qc

    def _prob1(self, counts: dict) -> float:
        """Convert shot counts to P(|1>) clamped to (0.01, 0.99)."""
        return float(np.clip(counts.get("1", 0) / self.shots, 0.01, 0.99))

    def sample_batch_full(self, n_particles: int, dims: int, beta: float) -> tuple:
        """
        Generate ALL random values for one QPSO iteration from quantum circuits.

        Builds 3 circuits per (particle x dimension): [H, H, RY], all submitted
        as a single batched job.

        Returns
        -------
        r1   : ndarray (n_particles, dims) — quantum cognitive weights
        r2   : ndarray (n_particles, dims) — quantum social weights
        u    : ndarray (n_particles, dims) — Beta-biased displacement magnitudes
        sign : ndarray (n_particles, dims) — quantum tunnelling direction +/-1
        """
        total = n_particles * dims
        circuits = []
        for _ in range(total):
            circuits.append(self._h_circuit())       # r1
            circuits.append(self._h_circuit())       # r2
            circuits.append(self._ry_circuit(beta))  # u + sign

        result = self.backend.run(circuits, shots=self.shots).result()

        r1_flat = np.zeros(total)
        r2_flat = np.zeros(total)
        u_flat = np.zeros(total)
        sign_flat = np.ones(total)

        for i in range(total):
            base = i * 3
            r1_flat[i] = self._prob1(result.get_counts(base))
            r2_flat[i] = self._prob1(result.get_counts(base + 1))
            ry_prob = self._prob1(result.get_counts(base + 2))
            u_flat[i] = ry_prob
            sign_flat[i] = 1.0 if ry_prob >= 0.5 else -1.0

        return (
            r1_flat.reshape(n_particles, dims),
            r2_flat.reshape(n_particles, dims),
            u_flat.reshape(n_particles, dims),
            sign_flat.reshape(n_particles, dims),
        )

    def sample_signs_only(self, n_particles: int, dims: int, beta: float) -> np.ndarray:
        """
        Generate quantum tunnelling signs (+1 or -1) for all particles.

        One RY circuit per (particle x dimension), all batched into one job.
        P(|1>) = sin^2(beta*pi/2) — Beta shapes the sign distribution
        directly through the quantum circuit rotation angle.

        Returns
        -------
        signs : ndarray (n_particles, dims) containing +/-1 values
        """
        total = n_particles * dims
        circuits = [self._ry_circuit(beta) for _ in range(total)]

        result = self.backend.run(circuits, shots=self.shots).result()

        signs = np.ones(total)
        for i in range(total):
            counts = result.get_counts(i)
            prob_one = counts.get("1", 0) / self.shots
            signs[i] = 1.0 if prob_one >= 0.5 else -1.0

        return signs.reshape(n_particles, dims)
