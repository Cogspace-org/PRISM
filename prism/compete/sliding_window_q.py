from __future__ import annotations

from collections import deque

import numpy as np

from .base import CompeteAgent


class SlidingWindowQAgent(CompeteAgent):
    """Q-learning with a finite replay buffer (sliding window).

    Old transitions fall out of the buffer, so Q-values for changed
    regions are no longer reinforced and decay through new experience.
    Replay from the buffer keeps recent knowledge fresh.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.10,
        window: int = 500,
        replay_per_step: int = 5,
        seed: int = 42,
    ) -> None:
        self.Q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_per_step = replay_per_step
        self.rng = np.random.default_rng(seed)

        self.buffer: deque[tuple[int, int, int, float, bool]] = deque(maxlen=window)

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(
        self, s: int, a: int, s_next: int, reward: float, done: bool
    ) -> None:
        self.buffer.append((s, a, s_next, reward, done))

        # direct update
        target = reward if done else reward + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # replay from buffer
        buf_len = len(self.buffer)
        if buf_len <= 1:
            return
        n_replay = min(self.replay_per_step, buf_len - 1)
        indices = self.rng.choice(buf_len, size=n_replay, replace=False)
        for idx in indices:
            sb, ab, snb, rb, db = self.buffer[idx]
            t = rb if db else rb + self.gamma * np.max(self.Q[snb])
            self.Q[sb, ab] += self.alpha * (t - self.Q[sb, ab])
