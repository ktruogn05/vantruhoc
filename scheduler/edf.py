"""EDF baseline - resume paused work, then promote earliest deadlines."""

from core.types import Request, SchedulerAction
from scheduler.base import BaseScheduler
from scheduler.utils import select_promote_action, select_resume_action


class EDFScheduler(BaseScheduler):
    """Earliest Deadline First baseline."""

    def select_action(self, state_snapshot: dict) -> SchedulerAction:
        resume = select_resume_action(state_snapshot)
        if resume is not None:
            return resume

        queue: list[Request] = state_snapshot["queue"]
        sorted_indices = sorted(range(len(queue)), key=lambda i: queue[i].deadline)
        return select_promote_action(state_snapshot, sorted_indices)
