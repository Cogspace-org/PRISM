from __future__ import annotations

import numpy as np

from .base import CompeteAgent


class RMaxAgent(CompeteAgent):
    """R-max: optimistic exploration via count-based initialization.

    State-action pairs visited fewer than *m* times are assigned the
    optimistic value R_max / (1 - gamma).  Once the count threshold is
    reached, standard Q-learning updates take over.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        m: int = 5,
        r_max: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.m = m
        self.r_max = r_max
        self.rng = np.random.default_rng(seed)

        self.v_max = r_max / (1.0 - gamma)
        self.Q = np.full((n_states, n_actions), self.v_max)
        self.count = np.zeros((n_states, n_actions), dtype=int)

    def select_action(self, state: int) -> int:
        # greedy — optimism drives exploration
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(
        self, s: int, a: int, s_next: int, reward: float, done: bool
    ) -> None:
        self.count[s, a] += 1
        if self.count[s, a] == self.m:
            # transition from optimistic to learned: reset to 0
            self.Q[s, a] = 0.0
        if self.count[s, a] >= self.m:
            target = reward if done else reward + self.gamma * np.max(self.Q[s_next])
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
