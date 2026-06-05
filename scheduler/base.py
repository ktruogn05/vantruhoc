"""Base scheduler protocol — shared by baselines and RL agent."""

from abc import ABC, abstractmethod

from core.types import SchedulerAction


class BaseScheduler(ABC):
    """All schedulers implement this interface.

    Contract:
    - Receives raw state_snapshot dict from env
    - Returns SchedulerAction struct
    - Env is agnostic to scheduler type
    """

    @abstractmethod
    def select_action(self, state_snapshot: dict) -> SchedulerAction:
        """Core decision. Must be implemented."""
        ...

    def on_episode_start(self) -> None:
        """Optional hook for stateful schedulers."""
        pass

    def on_episode_end(self, metrics: dict) -> None:
        """Optional hook for logging/learning."""
        pass
