"""Stable-Baselines3 wrappers for Q-learning / DQN baselines."""


class SB3Baseline:
    """Wrapper around SB3 agents for comparison experiments."""

    def __init__(self, env, algorithm="DQN", **kwargs):
        raise NotImplementedError("Phase 3")
