"""Core data models and enums for the RL pipeline."""

from enum import IntEnum
from dataclasses import dataclass, field

from core.constants import MAX_RESPONSE_TOKENS


class RequestStage(IntEnum):
    PREFILL = 0
    SWAP_IN_DEGRADED = 1
    DECODE = 2


class PreemptedLabel(IntEnum):
    RECOMPUTE_WAITING = 0
    SWAPPED_TO_CPU = 1


class PreemptStrategy(IntEnum):
    RECOMPUTE = 0
    SWAP = 1


class ActionType(IntEnum):
    NOOP = 0
    PROMOTE = 1
    PREEMPT = 2
    RESUME = 3


@dataclass
class Request:
    """Immutable arrival info + mutable processing state."""

    # --- Immutable (set at arrival) ---
    id: int
    prompt_tokens: int
    target_response_tokens: int
    priority: int               # 1, 2, 3
    arrival_time: int
    deadline: int

    # --- Mutable (updated during processing) ---
    stage: RequestStage | None = None
    tokens_generated: int = 0
    swap_in_remaining: int = 0
    cpu_tokens_held: int = 0
    preempted_label: PreemptedLabel | None = None
    ttft: int | None = None

    def __post_init__(self):
        self.target_response_tokens = min(
            self.target_response_tokens, MAX_RESPONSE_TOKENS
        )

    @property
    def total_tokens(self) -> int:
        """Current memory footprint: p_i + g_i."""
        return self.prompt_tokens + self.tokens_generated

    @property
    def is_complete(self) -> bool:
        return self.tokens_generated >= self.target_response_tokens


@dataclass
class SchedulerAction:
    """Structured action — output by all schedulers, consumed by env."""

    action_type: ActionType
    indices: list[int] = field(default_factory=list)
    preempt_strategy: PreemptStrategy = PreemptStrategy.RECOMPUTE


@dataclass
class StepMetrics:
    """Per-step metrics for logging/evaluation."""

    tokens_decoded: int = 0
    requests_completed: int = 0
    requests_aborted: int = 0
    sla_violations: int = 0
    oom_events: int = 0
    active_batch_size: int = 0
    queue_size: int = 0
    preempted_queue_size: int = 0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
