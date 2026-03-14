"""Tabular Q-Learning baseline for Exp C.

State = (position_index, direction) = 260 x 4 = 1040 states, 3 actions.
Actions are MiniGrid movement actions: turn_left (0), turn_right (1), forward (2).

Direction is needed because MiniGrid actions depend on facing direction:
Q(s, a) would be ambiguous with position-only states.
"""

import numpy as np


class TabularQLearning:
    """Tabular Q-Learning agent for MiniGrid FourRooms.

    Same API as BaseSRAgent: select_action(s, actions, agent_dir),
    update(s, s_next, reward).

    State encoding: (position_index, direction) -> flat index.
    """

    def __init__(self, n_states, state_mapper, alpha=0.1, gamma=0.95,
                 epsilon_start=0.3, epsilon_min=0.01, epsilon_decay=0.995,
                 seed=42, **kwargs):
        self.n_states = n_states  # position-only count (260)
        self.mapper = state_mapper
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_dirs = 4
        self.n_actions = 3
        self.n_joint_states = n_states * self.n_dirs  # 1040

        self.Q = np.zeros((self.n_joint_states, self.n_actions),
                          dtype=np.float64)
        self.visit_counts = np.zeros(n_states, dtype=np.int64)
        self.total_steps = 0
        self._rng = np.random.default_rng(seed)

        # Track last (dir, action) for update()
        self._last_dir = 0
        self._last_action = 0

    def _joint_index(self, pos_idx, direction):
        """Encode (position, direction) as a flat index."""
        return pos_idx * self.n_dirs + direction

    def _epsilon(self):
        """Current epsilon with exponential decay."""
        return max(self.epsilon_min,
                   self.epsilon_start * self.epsilon_decay ** self.total_steps)

    def select_action(self, s, available_actions, agent_dir):
        """Epsilon-greedy action selection.

        Args:
            s: Position state index (0-259).
            available_actions: List of valid action indices [0, 1, 2].
            agent_dir: Agent facing direction (0-3).

        Returns:
            Chosen action index.
        """
        self._last_dir = agent_dir

        if self._rng.random() < self._epsilon():
            action = int(self._rng.choice(available_actions))
        else:
            ji = self._joint_index(s, agent_dir)
            q_vals = self.Q[ji, available_actions]
            best = np.where(np.isclose(q_vals, q_vals.max()))[0]
            action = available_actions[self._rng.choice(best)]

        self._last_action = action
        return action

    def update(self, s, s_next, reward):
        """Q-Learning update: Q(s,a) += alpha * (r + gamma * max Q(s') - Q(s,a)).

        Uses last direction and action from select_action().
        For s_next, we compute max over all (direction, action) pairs at that position,
        but since we don't know the next direction yet, we use the standard
        approximation: max over actions at current observed next state.
        """
        ji = self._joint_index(s, self._last_dir)
        a = self._last_action

        # For next state, compute max Q across all directions and actions
        best_next = -np.inf
        for d in range(self.n_dirs):
            ji_next = self._joint_index(s_next, d)
            q_max = self.Q[ji_next].max()
            if q_max > best_next:
                best_next = q_max

        td_error = reward + self.gamma * best_next - self.Q[ji, a]
        self.Q[ji, a] += self.alpha * td_error

        self.visit_counts[s] += 1
        self.total_steps += 1

    def uncertainty(self, s):
        """Stub: constant uncertainty (Q-Learning has no uncertainty model)."""
        return 0.5

    def all_uncertainties(self):
        """Stub: constant uncertainty for all states."""
        return np.full(self.n_states, 0.5)

    def confidence(self, s):
        """Stub: constant confidence."""
        return 0.5

    def all_confidences(self):
        """Stub: constant confidence for all states."""
        return np.full(self.n_states, 0.5)
