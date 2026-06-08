"""Priority baseline with resume and capacity-aware preemption."""

from core.constants import CPU_RAM_MAX_TOKENS, GPU_VRAM_MAX_TOKENS
from core.types import (
    ActionType,
    PreemptStrategy,
    Request,
    RequestStage,
    SchedulerAction,
)
from scheduler.base import BaseScheduler
from scheduler.utils import select_promote_action, select_resume_action


class PriorityScheduler(BaseScheduler):
    """Promote high priority requests and preempt low priority work under pressure."""

    def select_action(self, state_snapshot: dict) -> SchedulerAction:
        resume = select_resume_action(state_snapshot)
        if resume is not None:
            return resume

        batch: list[Request] = state_snapshot["active_batch"]
        queue: list[Request] = state_snapshot["queue"]
        gpu_used: int = state_snapshot["gpu_tokens_used"]
        cpu_used: int = state_snapshot["cpu_tokens_used"]

        gpu_util = gpu_used / GPU_VRAM_MAX_TOKENS
        if gpu_util > 0.9 and queue:
            max_queue_prio = max(r.priority for r in queue)
            decode_reqs = [
                (i, r)
                for i, r in enumerate(batch)
                if r.stage == RequestStage.DECODE
            ]
            if decode_reqs:
                min_idx, min_req = min(decode_reqs, key=lambda x: x[1].priority)
                if max_queue_prio > min_req.priority:
                    strategy = (
                        PreemptStrategy.SWAP
                        if cpu_used + min_req.total_tokens <= CPU_RAM_MAX_TOKENS
                        else PreemptStrategy.RECOMPUTE
                    )
                    return SchedulerAction(
                        action_type=ActionType.PREEMPT,
                        indices=[min_idx],
                        preempt_strategy=strategy,
                    )

        sorted_indices = sorted(
            range(len(queue)),
            key=lambda i: (-queue[i].priority, queue[i].arrival_time),
        )
        return select_promote_action(state_snapshot, sorted_indices)
