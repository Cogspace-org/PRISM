from __future__ import annotations

import numpy as np

from .base import CompeteAgent


class DynaQAgent(CompeteAgent):
    """Dyna-Q: Q-learning + learned transition model + planning replay.

    Optional Dyna-Q+ mode adds a recency bonus kappa * sqrt(tau) to
    encourage revisiting transitions that haven't been seen recently.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.10,
        n_planning: int = 10,
        kappa: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.Q = np.zeros((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.kappa = kappa  # >0 enables Dyna-Q+
        self.rng = np.random.default_rng(seed)

        # model: (s, a) -> (s_next, reward)
        self.model: dict[tuple[int, int], tuple[int, float]] = {}
        self.observed_sa: list[tuple[int, int]] = []

        # time tracking for Dyna-Q+ bonus
        self._step_count = 0
        self._last_visit: dict[tuple[int, int], int] = {}

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(
        self, s: int, a: int, s_next: int, reward: float, done: bool
    ) -> None:
        self._step_count += 1

        # --- direct Q update ---
        target = reward if done else reward + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # --- model update ---
        if (s, a) not in self.model:
            self.observed_sa.append((s, a))
        self.model[(s, a)] = (s_next, reward)
        self._last_visit[(s, a)] = self._step_count

        # --- planning ---
        n_obs = len(self.observed_sa)
        if n_obs == 0:
            return
        for _ in range(self.n_planning):
            idx = int(self.rng.integers(n_obs))
            sp, ap = self.observed_sa[idx]
            snp, rp = self.model[(sp, ap)]
            bonus = 0.0
            if self.kappa > 0:
                tau = self._step_count - self._last_visit.get((sp, ap), 0)
                bonus = self.kappa * np.sqrt(tau)
            target_p = rp + bonus + self.gamma * np.max(self.Q[snp])
            self.Q[sp, ap] += self.alpha * (target_p - self.Q[sp, ap])
