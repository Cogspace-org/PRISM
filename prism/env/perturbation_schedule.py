"""Configurable perturbation schedules for each experiment.

References:
    master.md section 6 (experiment protocols)
"""

from dataclasses import dataclass, field


@dataclass
class PerturbationEvent:
    """A single scheduled perturbation."""
    episode: int
    ptype: str
    kwargs: dict = field(default_factory=dict)


class PerturbationSchedule:
    """Ordered sequence of perturbation events."""

    def __init__(self, events: list[PerturbationEvent]):
        self.events = sorted(events, key=lambda e: e.episode)
        self._index = 0

    def reset(self):
        """Reset schedule pointer to the beginning."""
        self._index = 0

    def get_events_for_episode(self, episode: int) -> list[PerturbationEvent]:
        """Return all events scheduled at this episode number.

        Uses a sorted pointer for O(1) amortized lookup. Call reset()
        before re-iterating from episode 0.
        """
        triggered = []
        while (self._index < len(self.events)
               and self.events[self._index].episode <= episode):
            if self.events[self._index].episode == episode:
                triggered.append(self.events[self._index])
            self._index += 1
        return triggered

    def phase_boundaries(self) -> list[int]:
        """Return the episode numbers where perturbations occur."""
        return [e.episode for e in self.events]

    @classmethod
    def exp_a(cls, phase_1_episodes=200, phase_2_episodes=200,
              goal_positions=None) -> "PerturbationSchedule":
        """Schedule for Experiment A (calibration).

        3 phases of reward_shift perturbations:
            Phase 1: [0, phase_1_episodes) — stable learning at goal_positions[0]
            Phase 2: [phase_1_episodes, phase_1+phase_2) — goal shifts to [1]
            Phase 3: [phase_1+phase_2, ...) — goal shifts to [2]

        Args:
            phase_1_episodes: Number of episodes in phase 1.
            phase_2_episodes: Number of episodes in phase 2.
            goal_positions: List of 3 (x, y) tuples, one per phase.
        """
        if goal_positions is None or len(goal_positions) < 3:
            raise ValueError("goal_positions must have at least 3 entries")

        events = [
            PerturbationEvent(
                episode=phase_1_episodes,
                ptype="reward_shift",
                kwargs={"new_goal_pos": tuple(goal_positions[1])},
            ),
            PerturbationEvent(
                episode=phase_1_episodes + phase_2_episodes,
                ptype="reward_shift",
                kwargs={"new_goal_pos": tuple(goal_positions[2])},
            ),
        ]
        return cls(events)

    @classmethod
    def exp_c(cls, stable_episodes=200, r_perturb_episodes=100,
              restab_episodes=100, m_perturb_episodes=100,
              final_episodes=100, goal_positions=None,
              wall_pos=None, wall_positions=None) -> "PerturbationSchedule":
        """Schedule for Experiment C (adaptation / R-M asymmetry).

        5 phases:
            Phase 1: [0, stable)                         — stable learning
            Phase 2: [stable, stable+r_perturb)          — reward_shift
            Phase 3: [stable+r_perturb, stable+r+restab) — restabilisation
            Phase 4: [stable+r+restab, +m_perturb)       — door_block
            Phase 5: [..., end)                          — final restabilisation

        Args:
            stable_episodes: Phase 1 length.
            r_perturb_episodes: Phase 2 length.
            restab_episodes: Phase 3 length.
            m_perturb_episodes: Phase 4 length.
            final_episodes: Phase 5 length.
            goal_positions: List of 2 (x, y) — initial and shifted goal.
            wall_pos: (x, y) of single passage to block (backward compat).
            wall_positions: List of (x, y) passages to block (overrides wall_pos).
        """
        if goal_positions is None or len(goal_positions) < 2:
            raise ValueError("goal_positions must have at least 2 entries")

        # Determine wall positions to block
        if wall_positions is not None:
            walls = [tuple(w) for w in wall_positions]
        elif wall_pos is not None:
            walls = [tuple(wall_pos)]
        else:
            raise ValueError("wall_pos or wall_positions is required for exp_c")

        ep_r = stable_episodes
        ep_m = stable_episodes + r_perturb_episodes + restab_episodes

        events = [
            PerturbationEvent(
                episode=ep_r,
                ptype="reward_shift",
                kwargs={"new_goal_pos": tuple(goal_positions[1])},
            ),
        ]
        # Create one door_block event per wall position, all at same episode
        for w in walls:
            events.append(
                PerturbationEvent(
                    episode=ep_m,
                    ptype="door_block",
                    kwargs={"wall_pos": w},
                )
            )
        return cls(events)
