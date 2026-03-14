"""Two-room grid environment for Exp E (R/M asymmetry demonstration).

9x9 grid with a vertical wall at col=4. Permanent walls at col=4 for
rows that will NEVER be opened. Switchable passage positions (initially
blocked or open) are part of the state space and can be toggled.

Default layout: permanent walls at rows 0,1,3,5,6,7,8 col=4 (7 walls).
Position (4,4) starts open (passage). Position (2,4) starts blocked.
n_states = 74 (constant across perturbations).

Layout (initial state):
    . . . . # . . . .     # = permanent wall (not a state)
    . . . . # . . . .     B = blocked (state, but impassable)
    . . . . B . . . .     . = open (accessible state)
    . . . . # . . . .
    . . . . . . . . .  <- passage at row=4, col=4
    . . . . # . . . .
    . . . . # . . . .
    . . . . # . . . .
    . . . . # . . . .

Perturbation example (Phase 4):
    block (4,4) + open (2,4) -> passage moves from row 4 to row 2.
    n_states remains 74. 73 accessible + 1 blocked.
"""

import numpy as np


class TwoRoomEnv:
    """Two-room grid world for Exp E.

    Actions: 0=up, 1=down, 2=left, 3=right (cardinal, no facing direction).
    Permanent walls are excluded from the state space.
    Switchable positions (passages) are part of the state space and can
    be blocked/opened via _blocked set. Blocked cells become self-loops.
    """

    def __init__(self, rows=9, cols=9, walls=None, blocked=None,
                 goal_pos=(0, 8), max_steps=200, gamma=0.95):
        """Initialize the two-room environment.

        Parameters
        ----------
        rows, cols : int
            Grid dimensions.
        walls : set of (row, col) or None
            Permanent wall positions (excluded from state space).
        blocked : set of (row, col) or None
            Initially blocked positions (in state space but impassable).
        goal_pos : (row, col)
            Goal position (reward = 1.0 on arrival).
        max_steps : int
            Maximum steps per episode.
        gamma : float
            Discount factor for SR computation.
        """
        self.rows = rows
        self.cols = cols
        self.walls = set(walls) if walls else set()
        self.goal_pos = goal_pos
        self.max_steps = max_steps
        self.gamma = gamma

        # Enumerate states: all cells except permanent walls
        self.states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    self.states.append((r, c))

        self._n_states = len(self.states)
        self._pos_to_idx = {pos: i for i, pos in enumerate(self.states)}
        self._idx_to_pos = {i: pos for i, pos in enumerate(self.states)}

        # Dynamically blocked cells (barrier)
        self._blocked = set(blocked) if blocked else set()

        # Actions: up, down, left, right
        self._actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Episode state
        self._current_state = 0
        self._step_count = 0
        self._done = False

        # Reward vector (1.0 at goal, 0 elsewhere)
        self._reward_vec = np.zeros(self._n_states)
        if goal_pos in self._pos_to_idx:
            self._reward_vec[self._pos_to_idx[goal_pos]] = 1.0

    @property
    def n_states(self):
        """Number of states (constant = rows * cols)."""
        return self._n_states

    def get_pos(self, idx):
        """Convert state index to (row, col) position."""
        return self._idx_to_pos[idx]

    def get_index(self, pos):
        """Convert (row, col) position to state index."""
        return self._pos_to_idx[pos]

    def reset(self, start=None, rng=None):
        """Reset environment to a starting state.

        Parameters
        ----------
        start : int or (row, col) or None
            Starting state. If None, random non-goal, non-blocked state.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        state : int
            Starting state index.
        """
        self._step_count = 0
        self._done = False

        if start is not None:
            if isinstance(start, int):
                self._current_state = start
            else:
                self._current_state = self._pos_to_idx[tuple(start)]
        else:
            if rng is None:
                rng = np.random.default_rng()
            # Pick random non-goal, non-blocked state
            valid = [i for i in range(self._n_states)
                     if self.states[i] != self.goal_pos
                     and self.states[i] not in self._blocked]
            self._current_state = rng.choice(valid)

        return self._current_state

    def step(self, state, action):
        """Take one step from state with action.

        Parameters
        ----------
        state : int
            Current state index.
        action : int
            Action index (0=up, 1=down, 2=left, 3=right).

        Returns
        -------
        next_state : int
            Next state index.
        reward : float
            Reward received.
        done : bool
            Whether episode ended (goal reached or max steps).
        """
        self._step_count += 1
        pos = self._idx_to_pos[state]

        # Blocked state -> self-loop
        if pos in self._blocked:
            next_state = state
        else:
            dr, dc = self._actions[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            next_pos = (nr, nc)

            # Boundary check, wall check, and blocked check
            if (0 <= nr < self.rows and 0 <= nc < self.cols
                    and next_pos not in self.walls
                    and next_pos not in self._blocked):
                next_state = self._pos_to_idx[next_pos]
            else:
                next_state = state  # bounce

        self._current_state = next_state
        next_pos = self._idx_to_pos[next_state]

        # Reward
        reward = 1.0 if next_pos == self.goal_pos else 0.0

        # Done conditions
        done = (next_pos == self.goal_pos) or (self._step_count >= self.max_steps)
        self._done = done

        return next_state, reward, done

    def get_neighbors(self, state):
        """Get result of each action from a state.

        Parameters
        ----------
        state : int
            State index.

        Returns
        -------
        neighbors : list of (action, next_state)
            Each entry is (action_index, resulting_state_index).
            Includes self-loops from bounces and blocked states.
        """
        pos = self._idx_to_pos[state]
        neighbors = []

        if pos in self._blocked:
            # Blocked state: all actions lead to self
            for a in range(4):
                neighbors.append((a, state))
            return neighbors

        for a in range(4):
            dr, dc = self._actions[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            next_pos = (nr, nc)

            if (0 <= nr < self.rows and 0 <= nc < self.cols
                    and next_pos not in self.walls
                    and next_pos not in self._blocked):
                neighbors.append((a, self._pos_to_idx[next_pos]))
            else:
                neighbors.append((a, state))  # bounce

        return neighbors

    def transition_matrix(self):
        """Compute one-step transition matrix T under uniform random policy.

        Returns
        -------
        T : ndarray (n_states, n_states)
            T[i, j] = P(s' = j | s = i) under uniform random policy.
        """
        n = self._n_states
        T = np.zeros((n, n))

        for i in range(n):
            neighbors = self.get_neighbors(i)
            for _, j in neighbors:
                T[i, j] += 0.25  # uniform over 4 actions

        return T

    def true_sr(self, gamma=None):
        """Compute analytical SR matrix M* = (I - gamma * T)^{-1}.

        Parameters
        ----------
        gamma : float or None
            Discount factor. Uses self.gamma if None.

        Returns
        -------
        M_star : ndarray (n_states, n_states)
        """
        if gamma is None:
            gamma = self.gamma
        T = self.transition_matrix()
        I = np.eye(self._n_states)
        return np.linalg.inv(I - gamma * T)

    def reward_vector(self):
        """Return reward vector R (1.0 at goal, 0 elsewhere).

        Returns
        -------
        R : ndarray (n_states,)
        """
        return self._reward_vec.copy()

    def apply_perturbation(self, ptype, **kwargs):
        """Apply a perturbation to the environment.

        Parameters
        ----------
        ptype : str
            "reward_shift" : move goal to new_goal_pos=(row, col)
            "block_passage" : add pos=(row, col) to blocked set
            "open_passage" : remove pos=(row, col) from blocked set
        """
        if ptype == "reward_shift":
            new_goal = kwargs["new_goal_pos"]
            self._reward_vec[:] = 0.0
            self.goal_pos = tuple(new_goal)
            if self.goal_pos in self._pos_to_idx:
                self._reward_vec[self._pos_to_idx[self.goal_pos]] = 1.0

        elif ptype == "block_passage":
            pos = tuple(kwargs["pos"])
            if pos not in self._pos_to_idx:
                raise ValueError(f"Position {pos} not in state space")
            self._blocked.add(pos)

        elif ptype == "open_passage":
            pos = tuple(kwargs["pos"])
            self._blocked.discard(pos)

        else:
            raise ValueError(f"Unknown perturbation type: {ptype}")

    def blocked_indices(self):
        """Return set of state indices that are currently blocked."""
        return {self._pos_to_idx[pos] for pos in self._blocked
                if pos in self._pos_to_idx}

    @classmethod
    def default(cls, gamma=0.95):
        """Create the standard 9x9 two-room layout.

        Permanent walls at col=4 for rows 0,1,3,5,6,7,8 (7 walls).
        (4,4) is the initial open passage.
        (2,4) starts blocked (switchable).
        n_states = 74. 73 accessible + 1 blocked.
        """
        # Permanent walls: positions that will NEVER be opened
        walls = set()
        for r in [0, 1, 3, 5, 6, 7, 8]:
            walls.add((r, 4))

        # (2,4) starts blocked (can be opened later)
        blocked = {(2, 4)}

        return cls(rows=9, cols=9, walls=walls, blocked=blocked,
                   goal_pos=(0, 8), gamma=gamma)
