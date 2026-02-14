"""Central configuration and default hyperparameters for PRISM.

All default values are from master.md sections 5.3 and 5.4.
"""

from dataclasses import dataclass, field


@dataclass
class SRConfig:
    """Successor Representation layer parameters."""
    gamma: float = 0.95
    alpha_M: float = 0.1
    alpha_R: float = 0.3
    epsilon: float = 0.1


@dataclass
class MetaSRConfig:
    """Meta-SR layer parameters."""
    buffer_size: int = 20
    U_prior: float = 0.8
    decay: float = 0.85
    beta: float = 10.0
    theta_C: float = 0.3
    theta_change: float = 0.5


@dataclass
class ControllerConfig:
    """Controller parameters."""
    epsilon_min: float = 0.01
    epsilon_max: float = 0.50
    lambda_explore: float = 0.5
    theta_idk: float = 0.3


@dataclass
class EnvConfig:
    """Environment configuration."""
    env_id: str = "MiniGrid-FourRooms-v0"
    max_steps: int = 500
    render_mode: str | None = None


@dataclass
class PRISMConfig:
    """Top-level configuration assembling all sub-configs."""
    sr: SRConfig = field(default_factory=SRConfig)
    meta_sr: MetaSRConfig = field(default_factory=MetaSRConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    seed: int = 42
