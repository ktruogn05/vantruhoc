"""Action decoder - integer action ID to SchedulerAction plus masking."""

from __future__ import annotations

from itertools import combinations

import gymnasium
import numpy as np

from codec.obs_encoder import K_PREEMPTED, K_QUEUE
from core.constants import (
    BATCH_MAX_SIZE,
    CPU_RAM_MAX_TOKENS,
    GPU_VRAM_MAX_TOKENS,
    MAX_CONCURRENT_PREFILL,
)
from core.types import (
    ActionType,
    PreemptStrategy,
    PreemptedLabel,
    Request,
    RequestStage,
    SchedulerAction,
)


def _build_action_table() -> list[SchedulerAction]:
    """Pre-compute all discrete actions."""
    table: list[SchedulerAction] = [SchedulerAction(action_type=ActionType.NOOP)]

    for r in (1, 2):
        for combo in combinations(range(K_QUEUE), r):
            table.append(
                SchedulerAction(action_type=ActionType.PROMOTE, indices=list(combo))
            )

    for r in (1, 2):
        for combo in combinations(range(BATCH_MAX_SIZE), r):
            for strat in PreemptStrategy:
                table.append(
                    SchedulerAction(
                        action_type=ActionType.PREEMPT,
                        indices=list(combo),
                        preempt_strategy=strat,
                    )
                )

    for r in (1, 2):
        for combo in combinations(range(K_PREEMPTED), r):
            table.append(
                SchedulerAction(action_type=ActionType.RESUME, indices=list(combo))
            )

    return table


class ActionDecoder:
    """Maps Discrete(N) action index to structured SchedulerAction."""

    def __init__(self):
        self._table = _build_action_table()
        self.N_ACTIONS = len(self._table)
        self._action_space = gymnasium.spaces.Discrete(self.N_ACTIONS)

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return self._action_space

    def decode(self, action_id: int) -> SchedulerAction:
        return self._table[action_id]

    def get_action_mask(self, state: dict) -> np.ndarray:
        """Return bool array shape (N_ACTIONS,). True means valid action."""
        mask = np.zeros(self.N_ACTIONS, dtype=bool)

        batch: list[Request] = state["active_batch"]
        queue: list[Request] = state["queue"]
        preempted: list[Request] = state["preempted_queue"]
        gpu_free = GPU_VRAM_MAX_TOKENS - state["gpu_tokens_used"]
        cpu_free = CPU_RAM_MAX_TOKENS - state["cpu_tokens_used"]

        batch_slots_free = BATCH_MAX_SIZE - len(batch)
        prefill_count = sum(1 for r in batch if r.stage == RequestStage.PREFILL)

        for aid, action in enumerate(self._table):
            if action.action_type == ActionType.NOOP:
                mask[aid] = True
            elif action.action_type == ActionType.PROMOTE:
                mask[aid] = _can_promote(
                    action.indices,
                    queue,
                    batch_slots_free,
                    prefill_count,
                    gpu_free,
                )
            elif action.action_type == ActionType.PREEMPT:
                mask[aid] = _can_preempt(action, batch, cpu_free)
            elif action.action_type == ActionType.RESUME:
                mask[aid] = _can_resume(
                    action.indices,
                    preempted,
                    batch_slots_free,
                    prefill_count,
                    gpu_free,
                )

        return mask


def _can_promote(
    indices: list[int],
    queue: list[Request],
    batch_slots_free: int,
    prefill_count: int,
    gpu_free: int,
) -> bool:
    if not _indices_in_range(indices, len(queue)):
        return False
    if batch_slots_free < len(indices):
        return False
    if prefill_count + len(indices) > MAX_CONCURRENT_PREFILL:
        return False
    return sum(queue[idx].prompt_tokens for idx in indices) <= gpu_free


def _can_preempt(action: SchedulerAction, batch: list[Request], cpu_free: int) -> bool:
    if not _indices_in_range(action.indices, len(batch)):
        return False
    requests = [batch[idx] for idx in action.indices]
    if any(req.stage == RequestStage.PREFILL for req in requests):
        return False
    if action.preempt_strategy == PreemptStrategy.RECOMPUTE:
        return True
    return sum(req.total_tokens for req in requests) <= cpu_free


def _can_resume(
    indices: list[int],
    preempted: list[Request],
    batch_slots_free: int,
    prefill_count: int,
    gpu_free: int,
) -> bool:
    if not _indices_in_range(indices, len(preempted)):
        return False
    if batch_slots_free < len(indices):
        return False

    total_gpu_needed = 0
    extra_prefills = 0
    for idx in indices:
        req = preempted[idx]
        if req.preempted_label == PreemptedLabel.RECOMPUTE_WAITING:
            total_gpu_needed += req.prompt_tokens
            extra_prefills += 1
        elif req.preempted_label == PreemptedLabel.SWAPPED_TO_CPU:
            total_gpu_needed += req.total_tokens
        else:
            return False

    if prefill_count + extra_prefills > MAX_CONCURRENT_PREFILL:
        return False
    return total_gpu_needed <= gpu_free


def _indices_in_range(indices: list[int], size: int) -> bool:
    return bool(indices) and all(0 <= idx < size for idx in indices)
