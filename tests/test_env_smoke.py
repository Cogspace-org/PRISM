"""Phase 0 smoke test: verify MiniGrid FourRooms works as expected."""

import gymnasium as gym
import minigrid  # noqa: F401 -- registers MiniGrid envs in gymnasium
import pytest


class TestFourRoomsSmokeTest:
    """Basic smoke tests for MiniGrid FourRooms."""

    def test_env_creates(self):
        env = gym.make("MiniGrid-FourRooms-v0")
        obs, info = env.reset(seed=42)
        assert obs is not None
        env.close()

    def test_observation_has_expected_keys(self, four_rooms_env):
        obs, info = four_rooms_env.reset(seed=42)
        assert "image" in obs
        assert "direction" in obs

    def test_action_space_is_discrete_7(self, four_rooms_env):
        assert four_rooms_env.action_space.n == 7

    def test_grid_is_19x19(self, four_rooms_env):
        grid = four_rooms_env.unwrapped.grid
        assert grid.width == 19
        assert grid.height == 19

    def test_agent_pos_accessible(self, four_rooms_env):
        pos = four_rooms_env.unwrapped.agent_pos
        assert pos is not None
        assert len(pos) == 2
        x, y = int(pos[0]), int(pos[1])
        assert 0 < x < 18
        assert 0 < y < 18

    def test_agent_pos_updates_on_step(self, four_rooms_env):
        initial_pos = tuple(four_rooms_env.unwrapped.agent_pos)
        moved = False
        for _ in range(20):
            four_rooms_env.step(2)  # forward
            four_rooms_env.step(0)  # turn left
            four_rooms_env.step(2)  # forward
            new_pos = tuple(four_rooms_env.unwrapped.agent_pos)
            if new_pos != initial_pos:
                moved = True
                break
        assert moved, "Agent position should change after movement actions"

    def test_accessible_cells_in_expected_range(self, four_rooms_env):
        """FourRooms 19x19 should have roughly 100-150 accessible cells."""
        grid = four_rooms_env.unwrapped.grid
        accessible = 0
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is None or (cell is not None and cell.type in ("door", "goal")):
                    accessible += 1
        assert 80 <= accessible <= 300, f"Expected 80-300 accessible cells, got {accessible}"

    def test_deterministic_with_same_seed(self):
        """Same seed should produce the same grid layout."""
        env1 = gym.make("MiniGrid-FourRooms-v0")
        env2 = gym.make("MiniGrid-FourRooms-v0")

        env1.reset(seed=42)
        env2.reset(seed=42)

        pos1 = tuple(env1.unwrapped.agent_pos)
        pos2 = tuple(env2.unwrapped.agent_pos)

        assert pos1 == pos2

        env1.close()
        env2.close()
