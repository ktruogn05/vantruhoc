"""FCFS baseline — always promote oldest requests, never preempt."""

from core.constants import BATCH_MAX_SIZE, GPU_VRAM_MAX_TOKENS, MAX_CONCURRENT_PREFILL
from core.types import ActionType, Request, RequestStage, SchedulerAction
from scheduler.base import BaseScheduler


class FCFSScheduler(BaseScheduler):
    """First-Come First-Served. Promote from front of queue when possible."""

    def select_action(self, state_snapshot: dict) -> SchedulerAction:
        batch: list[Request] = state_snapshot["active_batch"]
        queue: list[Request] = state_snapshot["queue"]
        gpu_used: int = state_snapshot["gpu_tokens_used"]

        batch_free = BATCH_MAX_SIZE - len(batch)
        if batch_free <= 0 or not queue:
            return SchedulerAction(action_type=ActionType.NOOP)

        prefill_count = sum(
            1 for r in batch if r.stage == RequestStage.PREFILL
        )

        gpu_free = GPU_VRAM_MAX_TOKENS - gpu_used
        indices = []

        for i, req in enumerate(queue):
            if len(indices) >= 2:
                break
            if len(batch) + len(indices) >= BATCH_MAX_SIZE:
                break
            if prefill_count + len(indices) >= MAX_CONCURRENT_PREFILL:
                break
            if req.prompt_tokens <= gpu_free:
                indices.append(i)
                gpu_free -= req.prompt_tokens

        if not indices:
            return SchedulerAction(action_type=ActionType.NOOP)

        return SchedulerAction(action_type=ActionType.PROMOTE, indices=indices)
