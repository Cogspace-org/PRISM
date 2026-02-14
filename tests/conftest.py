"""Shared pytest fixtures for PRISM tests."""

import pytest
import gymnasium as gym
import minigrid  # noqa: F401 -- registers MiniGrid envs in gymnasium


@pytest.fixture
def four_rooms_env():
    """Create a fresh FourRooms environment with a fixed seed."""
    env = gym.make("MiniGrid-FourRooms-v0")
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def four_rooms_grid(four_rooms_env):
    """Return the unwrapped grid from FourRooms."""
    return four_rooms_env.unwrapped.grid
