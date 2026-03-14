"""Four-room grid environment for PRISM_compete (FourRooms benchmark).

13x13 grid with cross-shaped walls forming 4 rooms. Horizontal wall at
row=6, vertical wall at col=6. Each wall segment has one passage (open)
and one alternate position (blocked, switchable).

n_states = 152 (constant across perturbations).

Default layout:
     0  1  2  3  4  5  6  7  8  9  10 11 12
  0  .  .  .  .  .  .  #  .  .  .  .  .  G
  1  .  .  .  .  .  .  #  .  .  .  .  .  .
  2  .  .  .  .  .  .  P  .  .  .  .  .  .   <- north passage (open)
  3  .  .  .  .  .  .  B  .  .  .  .  .  .   <- north alternate (blocked)
  4  .  .  .  .  .  .  #  .  .  .  .  .  .
  5  .  .  .  .  .  .  #  .  .  .  .  .  .
  6  #  #  P  B  #  #  #  #  #  B  P  #  #   <- horizontal wall
  7  .  .  .  .  .  .  #  .  .  .  .  .  .
  8  .  .  .  .  .  .  #  .  .  .  .  .  .
  9  .  .  .  .  .  .  P  .  .  .  .  .  .   <- south passage (open)
 10  .  .  .  .  .  .  B  .  .  .  .  .  .   <- south alternate (blocked)
 11  .  .  .  .  .  .  #  .  .  .  .  .  .
 12  .  .  .  .  .  .  #  .  .  .  .  .  .

  # = permanent wall (not in state space)
  P = passage (open, in state space)
  B = blocked alternate (in state space, self-loop)
  G = goal at (0, 12)

Four rooms:
  NW: rows 0-5, cols 0-5  (36 cells)
  NE: rows 0-5, cols 7-12 (36 cells)
  SW: rows 7-12, cols 0-5 (36 cells)
  SE: rows 7-12, cols 7-12 (36 cells)

Perturbation example (Phase 4 M-change):
  block_passage (6, 2) + open_passage (6, 3) -> west passage moves.
  block_passage (2, 6) + open_passage (3, 6) -> north passage moves.
"""

from prism.env.two_room_env import TwoRoomEnv


class FourRoomEnv(TwoRoomEnv):
    """13x13 four-room grid world for PRISM_compete.

    Inherits all logic from TwoRoomEnv. Only the default() layout differs.
    Actions: 0=up, 1=down, 2=left, 3=right (cardinal, no facing direction).
    """

    @classmethod
    def default(cls, gamma=0.95):
        """Create the standard 13x13 four-room layout.

        Permanent walls: cross at row=6 / col=6 (17 cells).
        Passages: 4 open + 4 blocked alternates (8 switchable cells).
        n_states = 169 - 17 = 152. Initially 148 accessible + 4 blocked.
        """
        walls = set()

        # Vertical wall at col=6 — north segment (rows 0-5)
        for r in [0, 1, 4, 5]:
            walls.add((r, 6))
        # passage (2, 6) = open, alternate (3, 6) = blocked

        # Vertical wall at col=6 — south segment (rows 7-12)
        for r in [7, 8, 11, 12]:
            walls.add((r, 6))
        # passage (9, 6) = open, alternate (10, 6) = blocked

        # Horizontal wall at row=6 — west segment (cols 0-5)
        for c in [0, 1, 4, 5]:
            walls.add((6, c))
        # passage (6, 2) = open, alternate (6, 3) = blocked

        # Horizontal wall at row=6 — east segment (cols 7-12)
        for c in [7, 8, 11, 12]:
            walls.add((6, c))
        # passage (6, 10) = open, alternate (6, 9) = blocked

        # Intersection
        walls.add((6, 6))

        # Initially blocked alternates
        blocked = {(3, 6), (10, 6), (6, 3), (6, 9)}

        return cls(rows=13, cols=13, walls=walls, blocked=blocked,
                   goal_pos=(0, 12), max_steps=500, gamma=gamma)
