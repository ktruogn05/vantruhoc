"""Workload generator — Poisson arrivals with configurable distributions."""

import numpy as np

from core.constants import MAX_RESPONSE_TOKENS
from core.types import Request


class WorkloadGenerator:
    """Sinh request theo Poisson process.

    Inject vào LLMEnvSimple. Tách riêng để swap distribution khi cần.
    """

    def __init__(self, arrival_rate: float = 0.06, seed: int = 42):
        self._arrival_rate = arrival_rate
        self._rng = np.random.default_rng(seed)
        self._next_id = 0

        # Default distributions
        self._prompt_len_range = (64, 1024)
        self._response_len_range = (32, MAX_RESPONSE_TOKENS)
        self._priority_weights = (0.5, 0.3, 0.2)  # P(1), P(2), P(3)
        self._deadline_slack_range = (50, 200)

    def set_distributions(
        self,
        prompt_len: tuple[int, int] | None = None,
        response_len: tuple[int, int] | None = None,
        priority_weights: tuple[float, float, float] | None = None,
        deadline_slack: tuple[int, int] | None = None,
    ) -> None:
        if prompt_len is not None:
            self._prompt_len_range = prompt_len
        if response_len is not None:
            self._response_len_range = response_len
        if priority_weights is not None:
            self._priority_weights = priority_weights
        if deadline_slack is not None:
            self._deadline_slack_range = deadline_slack

    def generate_arrivals(self, current_time: int) -> list[Request]:
        """Return requests arriving at current_time. Count ~ Poisson(rate)."""
        n = self._rng.poisson(self._arrival_rate)
        requests = []
        for _ in range(n):
            p = int(self._rng.integers(*self._prompt_len_range))
            o = int(self._rng.integers(*self._response_len_range))
            pr = int(self._rng.choice([1, 2, 3], p=self._priority_weights))
            slack = int(self._rng.integers(*self._deadline_slack_range))
            deadline = current_time + p + o + slack

            req = Request(
                id=self._next_id,
                prompt_tokens=p,
                target_response_tokens=o,
                priority=pr,
                arrival_time=current_time,
                deadline=deadline,
            )
            requests.append(req)
            self._next_id += 1
        return requests

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._next_id = 0
