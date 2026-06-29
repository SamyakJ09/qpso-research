"""
YAML configuration loader for experiment parameters.

Replaces the hardcoded USER SETTINGS blocks from the original files
with a single source of truth loaded from a config file.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PSOConfig:
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive_coef: float = 1.494
    social_coef: float = 1.494
    velocity_clamp: float = 0.2


@dataclass
class QPSOConfig:
    beta_start: float = 1.0
    beta_end: float = 0.5
    shots: int = 256
    momentum_decay: float = 0.7
    momentum_weight: float = 0.1


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    # Experiment settings
    benchmarks: list[str] = field(default_factory=lambda: ["sphere", "rastrigin", "rosenbrock"])
    dimensions: list[int] = field(default_factory=lambda: [2, 5])
    num_runs: int = 5
    num_particles: int = 30
    max_iterations: int = 150
    success_threshold: float = 1e-6
    stagnation_limit: int = 15
    seed: int = 42
    qpso_modes: list[str] = field(default_factory=lambda: ["math"])

    # Algorithm configs
    pso: PSOConfig = field(default_factory=PSOConfig)
    qpso: QPSOConfig = field(default_factory=QPSOConfig)

    # Output
    results_dir: str = "results"
    verbose: bool = True


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return ExperimentConfig()

    # Build nested configs
    pso_raw = raw.pop("pso", {})
    qpso_raw = raw.pop("qpso", {})

    pso_cfg = PSOConfig(**pso_raw) if pso_raw else PSOConfig()
    qpso_cfg = QPSOConfig(**qpso_raw) if qpso_raw else QPSOConfig()

    return ExperimentConfig(pso=pso_cfg, qpso=qpso_cfg, **raw)
