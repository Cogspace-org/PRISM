"""Adaptive controller using meta-SR signals for action selection.

References:
    master.md section 5.4
"""

import numpy as np


class PRISMController:
    """Policy controller integrating SR values with meta-SR uncertainty."""

    def __init__(self, sr_layer, meta_sr, epsilon_min: float = 0.01,
                 epsilon_max: float = 0.5, lambda_explore: float = 0.5):
        raise NotImplementedError("Phase 2")

    def select_action(self, s: int, available_actions: list) -> tuple:
        """Select action. Returns (action, confidence, idk_flag)."""
        raise NotImplementedError("Phase 2")
