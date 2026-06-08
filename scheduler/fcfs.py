"""FCFS baseline - resume oldest paused work, then promote oldest arrivals."""

from core.types import Request, SchedulerAction
from scheduler.base import BaseScheduler
from scheduler.utils import select_promote_action, select_resume_action


class FCFSScheduler(BaseScheduler):
    """First-Come First-Served baseline."""

    def select_action(self, state_snapshot: dict) -> SchedulerAction:
        resume = select_resume_action(state_snapshot)
        if resume is not None:
            return resume

        queue: list[Request] = state_snapshot["queue"]
        return select_promote_action(state_snapshot, list(range(len(queue))))
