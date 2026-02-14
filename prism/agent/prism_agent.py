"""Full PRISM agent assembling SR + Meta-SR + Controller.

References:
    master.md section 5
"""


class PRISMAgent:
    """Complete PRISM agent for MiniGrid environments."""

    def __init__(self, env, state_mapper, **kwargs):
        raise NotImplementedError("Phase 2")

    def train_episode(self) -> dict:
        """Run one episode. Returns metrics dict."""
        raise NotImplementedError("Phase 2")

    def train(self, n_episodes: int, log_every: int = 10):
        """Full training loop."""
        raise NotImplementedError("Phase 2")
