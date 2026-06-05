"""Reward calculator — stateless, weights configurable."""

from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Reward coefficients. Separated for easy tuning."""

    throughput: float = 1.0
    wait: float = 0.01
    sla: float = 10.0
    abort: float = 20.0
    recompute: float = 1.0
    oom: float = 10.0
    cpu_overflow: float = 50.0
    crash: float = 15.0


class RewardCalculator:
    """Stateless reward computation. Pure function on penalty inputs."""

    def __init__(self, weights: RewardWeights | None = None):
        self.w = weights or RewardWeights()

    def compute(
        self,
        tokens_decoded: int,
        wait_sum: float,
        sla_sum: float,
        abort_sum: float,
        recompute_sum: float,
        oom_count: int,
        cpu_overflow_count: int,
        crash_sum: float,
    ) -> float:
        """Compute scalar reward from per-step penalty accumulators."""
        r = self.w.throughput * tokens_decoded
        p = (
            self.w.wait * wait_sum
            + self.w.sla * sla_sum
            + self.w.abort * abort_sum
            + self.w.recompute * recompute_sum
            + self.w.oom * oom_count
            + self.w.cpu_overflow * cpu_overflow_count
            + self.w.crash * crash_sum
        )
        return r - p
