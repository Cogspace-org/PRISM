"""PRISM Phase 0 -- Environment verification script.

Verifies that MiniGrid FourRooms works correctly and extracts
the information needed for Phase 1 (SR implementation).

Run: python scripts/verify_env.py
"""

import sys

import gymnasium as gym
import minigrid  # noqa: F401 -- registers MiniGrid envs in gymnasium
import numpy as np


def verify_env_creation():
    """Test 1: Can we create the environment?"""
    print("=" * 60)
    print("TEST 1: Environment creation")
    print("=" * 60)

    env = gym.make("MiniGrid-FourRooms-v0")
    obs, info = env.reset(seed=42)

    print(f"  Environment ID: MiniGrid-FourRooms-v0")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action meanings: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Direction: {obs['direction']}")
    print()

    env.close()
    return True


def verify_grid_access():
    """Test 2: Can we access the internal grid and agent position?"""
    print("=" * 60)
    print("TEST 2: Grid and agent position access")
    print("=" * 60)

    env = gym.make("MiniGrid-FourRooms-v0")
    obs, info = env.reset(seed=42)

    unwrapped = env.unwrapped
    grid = unwrapped.grid
    agent_pos = unwrapped.agent_pos
    agent_dir = unwrapped.agent_dir

    print(f"  Grid dimensions: {grid.width} x {grid.height}")
    print(f"  Agent position: {tuple(agent_pos)}")
    print(f"  Agent direction: {agent_dir} (0=right, 1=down, 2=left, 3=up)")

    accessible = 0
    walls = 0
    doors = 0
    goals = 0
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is None:
                accessible += 1
            elif cell.type == "wall":
                walls += 1
            elif cell.type == "door":
                doors += 1
                accessible += 1
            elif cell.type == "goal":
                goals += 1
                accessible += 1

    print(f"  Accessible cells: {accessible}")
    print(f"  Walls: {walls}")
    print(f"  Doors: {doors}")
    print(f"  Goals: {goals}")
    print(f"  Total: {accessible + walls} (should equal {grid.width * grid.height})")
    print()

    env.close()
    return accessible


def verify_stepping():
    """Test 3: Can the agent take steps and do we get valid transitions?"""
    print("=" * 60)
    print("TEST 3: Agent stepping")
    print("=" * 60)

    env = gym.make("MiniGrid-FourRooms-v0")
    obs, info = env.reset(seed=42)

    positions_visited = set()
    total_reward = 0.0

    movement_actions = [0, 1, 2]  # left, right, forward
    rng = np.random.default_rng(42)
    for step in range(200):
        action = rng.choice(movement_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        pos = tuple(env.unwrapped.agent_pos)
        positions_visited.add(pos)
        total_reward += reward

        if step < 5:
            print(f"  Step {step}: action={action}, pos={pos}, "
                  f"reward={reward}, term={terminated}, trunc={truncated}")

        if terminated or truncated:
            obs, info = env.reset(seed=42 + step)

    print(f"  ...")
    print(f"  Unique positions visited (200 steps): {len(positions_visited)}")
    print(f"  Total reward: {total_reward}")
    print()

    env.close()
    return len(positions_visited)


def verify_grid_layout():
    """Test 4: Visualize the grid layout as ASCII."""
    print("=" * 60)
    print("TEST 4: Grid layout (ASCII)")
    print("=" * 60)

    env = gym.make("MiniGrid-FourRooms-v0")
    obs, info = env.reset(seed=42)

    grid = env.unwrapped.grid
    agent_pos = tuple(env.unwrapped.agent_pos)

    for y in range(grid.height):
        row = ""
        for x in range(grid.width):
            if (x, y) == agent_pos:
                row += "A "
            else:
                cell = grid.get(x, y)
                if cell is None:
                    row += ". "
                elif cell.type == "wall":
                    row += "# "
                elif cell.type == "door":
                    row += "D "
                elif cell.type == "goal":
                    row += "G "
                else:
                    row += "? "
        print(f"  {row}")

    print()
    env.close()
    return True


def verify_determinism():
    """Test 5: Same seed produces same grid layout."""
    print("=" * 60)
    print("TEST 5: Determinism (same seed = same layout)")
    print("=" * 60)

    positions_1 = []
    positions_2 = []

    for run, positions in [(1, positions_1), (2, positions_2)]:
        env = gym.make("MiniGrid-FourRooms-v0")
        obs, info = env.reset(seed=42)
        pos = tuple(env.unwrapped.agent_pos)
        positions.append(pos)

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(2)  # forward
            positions.append(tuple(env.unwrapped.agent_pos))
            if terminated or truncated:
                break
        env.close()

    match = positions_1 == positions_2
    print(f"  Run 1 positions: {positions_1[:5]}...")
    print(f"  Run 2 positions: {positions_2[:5]}...")
    print(f"  Deterministic: {match}")
    print()

    return match


def main():
    """Run all verification tests."""
    print()
    print("PRISM Phase 0 -- Environment Verification")
    print("=" * 60)
    print()

    results = {}

    try:
        results["creation"] = verify_env_creation()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["creation"] = False

    try:
        n_accessible = verify_grid_access()
        results["grid_access"] = n_accessible > 0
        results["n_accessible"] = n_accessible
    except Exception as e:
        print(f"  FAILED: {e}")
        results["grid_access"] = False

    try:
        n_visited = verify_stepping()
        results["stepping"] = n_visited > 1
    except Exception as e:
        print(f"  FAILED: {e}")
        results["stepping"] = False

    try:
        results["layout"] = verify_grid_layout()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["layout"] = False

    try:
        results["determinism"] = verify_determinism()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["determinism"] = False

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = all(v for k, v in results.items() if k != "n_accessible")

    for name, result in results.items():
        if name == "n_accessible":
            print(f"  Accessible states for SR matrix: {result}")
        else:
            status = "PASS" if result else "FAIL"
            print(f"  {name}: {status}")

    print()
    if all_pass:
        print("  All tests passed. Phase 0 environment verification complete.")
        print(f"  SR matrix will be {results.get('n_accessible', '?')} x {results.get('n_accessible', '?')}")
        print()
        print("  Next step: Phase 1 -- implement StateMapper and SRLayer")
    else:
        print("  Some tests FAILED. Fix issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
