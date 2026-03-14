from __future__ import annotations

from abc import ABC, abstractmethod


class CompeteAgent(ABC):
    """Common interface for all agents in the competition."""

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Return a cardinal action (0=up, 1=down, 2=left, 3=right)."""

    @abstractmethod
    def learn(
        self, s: int, a: int, s_next: int, reward: float, done: bool
    ) -> None:
        """Update after a single transition."""

    def on_episode_start(self) -> None:
        """Hook called at the beginning of each episode."""

    def on_episode_end(self, episode: int) -> None:
        """Hook called at the end of each episode."""

    def get_metrics(self) -> dict:
        """Return agent-specific metrics for the current timestep."""
        return {}
