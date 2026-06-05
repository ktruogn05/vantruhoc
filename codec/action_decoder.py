"""Action decoder — integer action ID → SchedulerAction struct + action masking."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import gymnasium

from core.constants import (
    BATCH_MAX_SIZE,
    GPU_VRAM_MAX_TOKENS,
    CPU_RAM_MAX_TOKENS,
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

# Layout constants (must match obs_encoder)
K_QUEUE = 16
K_PREEMPTED = 8


def _build_action_table() -> list[SchedulerAction]:
    """Pre-compute all possible actions at init time.

    Layout:
    [0]             : NoOp
    [1..136]        : Promote (C(16,1) + C(16,2) = 136)
    [137..168]      : Preempt (16 × 2 strategies = 32)
    [169..204]      : Resume (C(8,1) + C(8,2) = 36)
    Total: 205
    """
    table: list[SchedulerAction] = []

    # 0: NoOp
    table.append(SchedulerAction(action_type=ActionType.NOOP))

    # 1..136: Promote — pick 1 or 2 from K_QUEUE positions
    for r in (1, 2):
        for combo in combinations(range(K_QUEUE), r):
            table.append(
                SchedulerAction(action_type=ActionType.PROMOTE, indices=list(combo))
            )

    # 137..168: Preempt — pick 1 from BATCH_MAX_SIZE × 2 strategies
    for slot in range(BATCH_MAX_SIZE):
        for strat in PreemptStrategy:
            table.append(
                SchedulerAction(
                    action_type=ActionType.PREEMPT,
                    indices=[slot],
                    preempt_strategy=strat,
                )
            )

    # 169..204: Resume — pick 1 or 2 from K_PREEMPTED positions
    for r in (1, 2):
        for combo in combinations(range(K_PREEMPTED), r):
            table.append(
                SchedulerAction(action_type=ActionType.RESUME, indices=list(combo))
            )

    return table


class ActionDecoder:
    """Maps Discrete(N) action index ↔ SchedulerAction, with action masking."""

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
        """Return bool array shape (N_ACTIONS,). True = valid action."""
        mask = np.zeros(self.N_ACTIONS, dtype=bool)

        batch: list[Request] = state["active_batch"]
        queue: list[Request] = state["queue"]
        preempted: list[Request] = state["preempted_queue"]
        gpu_used: int = state["gpu_tokens_used"]
        cpu_used: int = state["cpu_tokens_used"]
        gpu_free = GPU_VRAM_MAX_TOKENS - gpu_used

        batch_size = len(batch)
        queue_size = len(queue)
        preempted_size = len(preempted)
        batch_slots_free = BATCH_MAX_SIZE - batch_size

        # Count current prefills in batch
        prefill_count = sum(
            1 for r in batch if r.stage == RequestStage.PREFILL
        )

        for aid, action in enumerate(self._table):
            if action.action_type == ActionType.NOOP:
                mask[aid] = True

            elif action.action_type == ActionType.PROMOTE:
                # All indices must be valid queue positions
                if all(idx < queue_size for idx in action.indices):
                    n_promote = len(action.indices)
                    if batch_slots_free >= n_promote:
                        if prefill_count + n_promote <= MAX_CONCURRENT_PREFILL:
                            # Check GPU has room for all prompts
                            total_prompt = sum(
                                queue[idx].prompt_tokens
                                for idx in action.indices
                            )
                            if total_prompt <= gpu_free:
                                mask[aid] = True

            elif action.action_type == ActionType.PREEMPT:
                idx = action.indices[0]
                if idx < batch_size:
                    req = batch[idx]
                    # Cannot preempt PREFILL
                    if req.stage != RequestStage.PREFILL:
                        if action.preempt_strategy == PreemptStrategy.RECOMPUTE:
                            mask[aid] = True
                        else:  # SWAP
                            # Check CPU capacity
                            if cpu_used + req.total_tokens <= CPU_RAM_MAX_TOKENS:
                                mask[aid] = True
                            else:
                                # Swap will fail with penalty, but is still a valid action
                                # (env handles gracefully). However masking it out is better
                                # to avoid wasted exploration.
                                pass

            elif action.action_type == ActionType.RESUME:
                if all(idx < preempted_size for idx in action.indices):
                    n_resume = len(action.indices)
                    if batch_slots_free >= n_resume:
                        # Check each request's requirements
                        total_gpu_needed = 0
                        extra_prefills = 0
                        valid = True
                        for idx in action.indices:
                            req = preempted[idx]
                            if req.preempted_label == PreemptedLabel.RECOMPUTE_WAITING:
                                total_gpu_needed += req.prompt_tokens
                                extra_prefills += 1
                            elif req.preempted_label == PreemptedLabel.SWAPPED_TO_CPU:
                                total_gpu_needed += req.total_tokens
                            else:
                                valid = False
                        if valid and total_gpu_needed <= gpu_free:
                            if prefill_count + extra_prefills <= MAX_CONCURRENT_PREFILL:
                                mask[aid] = True

        return mask
