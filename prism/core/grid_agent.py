"""Grid agents for cardinal-action environments (no MiniGrid facing direction).

Three agents for Exp E:
- GridSRAgent: SR-based with fixed epsilon (SR-Blind equivalent)
- GridPRISMAgent: SR + MetaSR + adaptive epsilon + exploration bonus
- GridQLearning: Model-free Q-learning baseline

All agents share the same API:
    select_action(s) -> action
    update(s, s_next, reward)

Actions are cardinal: 0=up, 1=down, 2=left, 3=right.
"""

import numpy as np
from prism.core.sr_layer import SRLayer
from prism.core.meta_sr import MetaSR


class GridSRAgent:
    """SR agent with fixed epsilon-greedy exploration on a cardinal grid.

    No facing direction, no MiniGrid. Uses env.get_neighbors() for
    action evaluation.

    Parameters
    ----------
    env : TwoRoomEnv
        Environment instance (needed for get_neighbors).
    gamma : float
        Discount factor.
    alpha_M : float
        SR learning rate.
    alpha_R : float
        Reward learning rate.
    epsilon : float
        Exploration rate (fixed).
    seed : int
        Random seed.
    """

    def __init__(self, env, gamma=0.95, alpha_M=0.1, alpha_R=0.3,
                 epsilon=0.1, seed=42):
        self.env = env
        self.n_states = env.n_states
        self.sr = SRLayer(self.n_states, gamma, alpha_M, alpha_R)
        self.visit_counts = np.zeros(self.n_states, dtype=np.int64)
        self.total_steps = 0
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    def _get_epsilon(self):
        """Return current exploration rate."""
        return self._epsilon

    def _exploration_value(self, s):
        """V_explore(s) = V(s) (no bonus for SR-Blind)."""
        return self.sr.value(s)

    def select_action(self, s):
        """Epsilon-greedy action selection based on V_explore of neighbors.

        Parameters
        ----------
        s : int
            Current state index.

        Returns
        -------
        action : int
            Chosen action (0-3).
        """
        epsilon = self._get_epsilon()

        if self._rng.random() < epsilon:
            return int(self._rng.integers(4))

        neighbors = self.env.get_neighbors(s)

        # Evaluate V_explore for each neighbor
        values = []
        for action, next_s in neighbors:
            values.append((action, self._exploration_value(next_s)))

        if not values:
            return int(self._rng.integers(4))

        best_val = max(v for _, v in values)
        best_actions = [a for a, v in values if np.isclose(v, best_val)]
        return int(self._rng.choice(best_actions))

    def update(self, s, s_next, reward):
        """Update SR and visit counts.

        Parameters
        ----------
        s : int
            Current state.
        s_next : int
            Next state.
        reward : float
            Reward received.
        """
        self.sr.update(s, s_next, reward)
        self.visit_counts[s] += 1
        self.total_steps += 1


class GridPRISMAgent(GridSRAgent):
    """PRISM agent for cardinal grid: SR + MetaSR + adaptive epsilon + bonus.

    Extends GridSRAgent with:
    - MetaSR for uncertainty tracking
    - Adaptive epsilon based on U(s)
    - Exploration bonus (count-based or uncertainty-based)

    Parameters
    ----------
    env : TwoRoomEnv
        Environment instance.
    gamma : float
        Discount factor.
    alpha_M, alpha_R : float
        SR learning rates.
    epsilon_min, epsilon_max : float
        Adaptive epsilon bounds.
    lambda_explore : float
        Exploration bonus weight.
    bonus_mode : str
        "count" or "uncertainty".
    buffer_size : int
        MetaSR buffer size.
    decay : float
        MetaSR uncertainty decay.
    beta : float
        MetaSR confidence sigmoid steepness.
    theta_change : float
        Change detection threshold.
    seed : int
        Random seed.
    """

    def __init__(self, env, gamma=0.95, alpha_M=0.1, alpha_R=0.3,
                 epsilon_min=0.05, epsilon_max=0.5,
                 lambda_explore=0.5, bonus_mode="count",
                 buffer_size=20, decay=0.85, beta=10.0,
                 theta_change=0.5, seed=42, adaptive_beta=False):
        super().__init__(env, gamma, alpha_M, alpha_R,
                         epsilon=epsilon_min, seed=seed)
        self._epsilon_min = epsilon_min
        self._epsilon_max = epsilon_max
        self.lambda_explore = lambda_explore
        self._bonus_mode = bonus_mode

        self.meta_sr = MetaSR(
            self.n_states,
            buffer_size=buffer_size,
            decay=decay,
            beta=beta,
            theta_change=theta_change,
            adaptive_beta=adaptive_beta,
        )

    def _get_epsilon(self):
        """Adaptive epsilon: not state-dependent for select_action,
        but we use mean U as proxy."""
        return self._epsilon_min

    def _adaptive_epsilon(self, s):
        """Compute state-dependent adaptive epsilon."""
        U_s = self.meta_sr.uncertainty(s)
        return self._epsilon_min + (self._epsilon_max - self._epsilon_min) * U_s

    def _exploration_bonus(self, s):
        """Compute exploration bonus for state s."""
        if self._bonus_mode == "count":
            visits = self.meta_sr.visit_counts[s]
            return 1.0 / np.sqrt(visits + 1)
        return self.meta_sr.uncertainty(s)

    def _exploration_value(self, s):
        """V_explore(s) = V(s) + lambda * bonus(s)."""
        return self.sr.value(s) + self.lambda_explore * self._exploration_bonus(s)

    def select_action(self, s):
        """Adaptive epsilon-greedy with exploration bonus."""
        epsilon = self._adaptive_epsilon(s)

        if self._rng.random() < epsilon:
            return int(self._rng.integers(4))

        neighbors = self.env.get_neighbors(s)
        values = []
        for action, next_s in neighbors:
            values.append((action, self._exploration_value(next_s)))

        if not values:
            return int(self._rng.integers(4))

        best_val = max(v for _, v in values)
        best_actions = [a for a, v in values if np.isclose(v, best_val)]
        return int(self._rng.choice(best_actions))

    def update(self, s, s_next, reward):
        """Update SR, MetaSR, and visit counts."""
        delta_M = self.sr.update(s, s_next, reward)
        self.meta_sr.observe(s, delta_M)
        self.visit_counts[s] += 1
        self.total_steps += 1

    def detect_change(self):
        """Detect structural change from recent uncertainty spike."""
        return self.meta_sr.detect_change()

    def all_uncertainties(self):
        """Return U for all states."""
        return self.meta_sr.all_uncertainties()

    def all_confidences(self):
        """Return C for all states."""
        return self.meta_sr.all_confidences()


class GridQLearning:
    """Tabular Q-learning baseline for cardinal grid.

    No SR, no direction. Q-table is (n_states, 4).

    Parameters
    ----------
    env : TwoRoomEnv
        Environment instance.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon_start : float
        Initial exploration rate.
    epsilon_min : float
        Minimum exploration rate.
    epsilon_decay : float
        Per-step epsilon decay.
    seed : int
        Random seed.
    """

    def __init__(self, env, alpha=0.1, gamma=0.95,
                 epsilon_start=0.3, epsilon_min=0.01,
                 epsilon_decay=0.995, seed=42):
        self.env = env
        self.n_states = env.n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((self.n_states, 4), dtype=np.float64)
        self.visit_counts = np.zeros(self.n_states, dtype=np.int64)
        self.total_steps = 0
        self._rng = np.random.default_rng(seed)
        self._last_action = 0

    def _epsilon(self):
        """Current epsilon with exponential decay."""
        return max(self.epsilon_min,
                   self.epsilon_start * self.epsilon_decay ** self.total_steps)

    def select_action(self, s):
        """Epsilon-greedy action selection.

        Parameters
        ----------
        s : int
            Current state index.

        Returns
        -------
        action : int
            Chosen action (0-3).
        """
        if self._rng.random() < self._epsilon():
            action = int(self._rng.integers(4))
        else:
            q_vals = self.Q[s]
            best = np.where(np.isclose(q_vals, q_vals.max()))[0]
            action = int(self._rng.choice(best))

        self._last_action = action
        return action

    def update(self, s, s_next, reward):
        """Q-learning update: Q(s,a) += alpha * (r + gamma * max Q(s') - Q(s,a)).

        Parameters
        ----------
        s : int
            Current state.
        s_next : int
            Next state.
        reward : float
            Reward received.
        """
        a = self._last_action
        best_next = self.Q[s_next].max()
        td_error = reward + self.gamma * best_next - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

        self.visit_counts[s] += 1
        self.total_steps += 1
