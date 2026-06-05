"""EDF baseline — promote requests with earliest deadline first."""

from core.constants import BATCH_MAX_SIZE, GPU_VRAM_MAX_TOKENS, MAX_CONCURRENT_PREFILL
from core.types import ActionType, Request, RequestStage, SchedulerAction
from scheduler.base import BaseScheduler


class EDFScheduler(BaseScheduler):
    """Earliest Deadline First. Promote most urgent requests."""

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

        # Sort queue indices by deadline (earliest first)
        sorted_indices = sorted(range(len(queue)), key=lambda i: queue[i].deadline)

        gpu_free = GPU_VRAM_MAX_TOKENS - gpu_used
        indices = []

        for i in sorted_indices:
            if len(indices) >= 2:
                break
            if len(batch) + len(indices) >= BATCH_MAX_SIZE:
                break
            if prefill_count + len(indices) >= MAX_CONCURRENT_PREFILL:
                break
            if queue[i].prompt_tokens <= gpu_free:
                indices.append(i)
                gpu_free -= queue[i].prompt_tokens

        if not indices:
            return SchedulerAction(action_type=ActionType.NOOP)

        return SchedulerAction(action_type=ActionType.PROMOTE, indices=indices)
