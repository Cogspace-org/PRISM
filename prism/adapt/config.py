"""Configuration dataclass for PRISM_adapt agents."""

from dataclasses import dataclass


@dataclass
class AdaptConfig:
    """Parameters for GridPRISMAdaptAgent.

    Inherits PRISM v1 defaults and adds adaptation-specific params.
    """

    # --- PRISM v1 inherited ---
    gamma: float = 0.95
    alpha_M: float = 0.1
    alpha_R: float = 0.3
    buffer_size: int = 20
    decay: float = 0.85
    beta: float = 10.0
    theta_change: float = 0.2
    bonus_mode: str = "count"

    # --- Adaptation-specific (M1/M2/M3) ---
    theta_reset: float = 0.20       # threshold for identifying states to reset
    lambda_recover: float = 0.5     # exploration bonus weight in recovery
    epsilon_max: float = 0.5        # epsilon for states in S_reset during recovery
    epsilon_min: float = 0.05       # epsilon outside recovery
    n_patience: int = 500           # steps without new change before exiting recovery
    detection_mode: str = "oracle"  # "oracle" (runner signals) or "auto" (MetaSR detect)
    min_steps_before_adapt: int = 11000  # warmup for auto detection mode
    max_triggers: int = 1           # max number of recovery triggers (1 = one-shot)

    # --- Learned precision (§3.0.1) ---
    adaptive_beta: bool = False     # β(s) = β₀ / (1 + var(buffer_s))

    # --- CUSUM auto-detection ---
    cusum_stat_method: str = "top_k_mean"   # "p90", "max", "top_k_mean"
    cusum_stat_k: int = 5                    # k for top_k_mean
    cusum_warmup: int = 300                  # skip first N episodes before baseline
    cusum_baseline_window: int = 100         # episodes to collect baseline (after warmup)
    cusum_slack_factor: float = 0.5          # k = slack_factor * sigma_0
    cusum_threshold_factor: float = 5.0      # h = threshold_factor * sigma_0
