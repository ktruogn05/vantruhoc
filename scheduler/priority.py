"""Priority baseline — promote highest priority first, preempt lowest on OOM pressure."""

from core.constants import BATCH_MAX_SIZE, GPU_VRAM_MAX_TOKENS, MAX_CONCURRENT_PREFILL
from core.types import (
    ActionType,
    PreemptStrategy,
    Request,
    RequestStage,
    SchedulerAction,
)
from scheduler.base import BaseScheduler


class PriorityScheduler(BaseScheduler):
    """Priority-based. Promote highest priority. Preempt lowest priority when
    GPU utilization is high and queue has higher-priority requests waiting."""

    def select_action(self, state_snapshot: dict) -> SchedulerAction:
        batch: list[Request] = state_snapshot["active_batch"]
        queue: list[Request] = state_snapshot["queue"]
        gpu_used: int = state_snapshot["gpu_tokens_used"]

        # Check if we should preempt: GPU > 90% full and queue has high-prio requests
        gpu_util = gpu_used / GPU_VRAM_MAX_TOKENS
        if gpu_util > 0.9 and queue:
            max_queue_prio = max(r.priority for r in queue)
            # Find lowest priority decode request in batch
            decode_reqs = [
                (i, r) for i, r in enumerate(batch)
                if r.stage == RequestStage.DECODE
            ]
            if decode_reqs:
                min_idx, min_req = min(decode_reqs, key=lambda x: x[1].priority)
                if max_queue_prio > min_req.priority:
                    return SchedulerAction(
                        action_type=ActionType.PREEMPT,
                        indices=[min_idx],
                        preempt_strategy=PreemptStrategy.SWAP,
                    )

        # Otherwise promote highest priority from queue
        batch_free = BATCH_MAX_SIZE - len(batch)
        if batch_free <= 0 or not queue:
            return SchedulerAction(action_type=ActionType.NOOP)

        prefill_count = sum(
            1 for r in batch if r.stage == RequestStage.PREFILL
        )

        # Sort by priority descending, then by arrival time ascending
        sorted_indices = sorted(
            range(len(queue)),
            key=lambda i: (-queue[i].priority, queue[i].arrival_time),
        )

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
