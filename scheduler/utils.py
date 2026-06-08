"""Shared helper decisions for non-RL schedulers."""

from core.constants import BATCH_MAX_SIZE, GPU_VRAM_MAX_TOKENS, MAX_CONCURRENT_PREFILL
from core.types import (
    ActionType,
    PreemptedLabel,
    Request,
    RequestStage,
    SchedulerAction,
)


def select_resume_action(state_snapshot: dict) -> SchedulerAction | None:
    """Resume the oldest preempted requests that fit current resources."""
    batch: list[Request] = state_snapshot["active_batch"]
    preempted: list[Request] = state_snapshot["preempted_queue"]
    gpu_free = GPU_VRAM_MAX_TOKENS - state_snapshot["gpu_tokens_used"]

    if not preempted or len(batch) >= BATCH_MAX_SIZE:
        return None

    prefill_count = sum(1 for r in batch if r.stage == RequestStage.PREFILL)
    indices = []
    for idx, req in enumerate(preempted):
        if len(indices) >= 2:
            break
        if len(batch) + len(indices) >= BATCH_MAX_SIZE:
            break

        if req.preempted_label == PreemptedLabel.RECOMPUTE_WAITING:
            if prefill_count + sum(
                1
                for i in indices
                if preempted[i].preempted_label
                == PreemptedLabel.RECOMPUTE_WAITING
            ) >= MAX_CONCURRENT_PREFILL:
                continue
            needed = req.prompt_tokens
        elif req.preempted_label == PreemptedLabel.SWAPPED_TO_CPU:
            needed = req.total_tokens
        else:
            continue

        if needed <= gpu_free:
            indices.append(idx)
            gpu_free -= needed

    if not indices:
        return None
    return SchedulerAction(action_type=ActionType.RESUME, indices=indices)


def select_promote_action(
    state_snapshot: dict,
    sorted_indices: list[int],
) -> SchedulerAction:
    """Promote the first queue requests in sorted_indices that fit."""
    batch: list[Request] = state_snapshot["active_batch"]
    queue: list[Request] = state_snapshot["queue"]
    gpu_free = GPU_VRAM_MAX_TOKENS - state_snapshot["gpu_tokens_used"]

    if not queue or len(batch) >= BATCH_MAX_SIZE:
        return SchedulerAction(action_type=ActionType.NOOP)

    prefill_count = sum(1 for r in batch if r.stage == RequestStage.PREFILL)
    indices = []
    for idx in sorted_indices:
        if len(indices) >= 2:
            break
        if len(batch) + len(indices) >= BATCH_MAX_SIZE:
            break
        if prefill_count + len(indices) >= MAX_CONCURRENT_PREFILL:
            break
        if idx < 0 or idx >= len(queue):
            continue
        if queue[idx].prompt_tokens <= gpu_free:
            indices.append(idx)
            gpu_free -= queue[idx].prompt_tokens

    if not indices:
        return SchedulerAction(action_type=ActionType.NOOP)
    return SchedulerAction(action_type=ActionType.PROMOTE, indices=indices)
