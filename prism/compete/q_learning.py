from __future__ import annotations

import numpy as np

from .base import CompeteAgent


class TabularQLearning(CompeteAgent):
    """Standard tabular Q-learning with fixed epsilon-greedy."""

    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.Q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # random tie-breaking
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(
        self, s: int, a: int, s_next: int, reward: float, done: bool
    ) -> None:
        target = reward if done else reward + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
