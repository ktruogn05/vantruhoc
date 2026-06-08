"""Workload generator - Poisson arrivals with configurable distributions."""

import numpy as np

from core.constants import MAX_RESPONSE_TOKENS
from core.types import Request


class WorkloadGenerator:
    """Generate request arrivals from a configurable Poisson process."""

    def __init__(
        self,
        arrival_rate: float = 0.06,
        seed: int | None = 42,
        arrival_horizon: int | None = None,
        max_arrivals: int | None = None,
    ):
        self._arrival_rate = arrival_rate
        self._rng = np.random.default_rng(seed)
        self._next_id = 0
        self._arrival_horizon = arrival_horizon
        self._max_arrivals = max_arrivals

        self._prompt_len_range = (64, 1024)
        self._response_len_range = (32, MAX_RESPONSE_TOKENS)
        self._priority_weights = (0.5, 0.3, 0.2)
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
        if self._arrival_horizon is not None and current_time > self._arrival_horizon:
            return []
        if self._max_arrivals is not None and self._next_id >= self._max_arrivals:
            return []

        n = int(self._rng.poisson(self._arrival_rate))
        if self._max_arrivals is not None:
            n = min(n, self._max_arrivals - self._next_id)

        requests = []
        for _ in range(n):
            p = int(self._rng.integers(*self._prompt_len_range))
            o = int(self._rng.integers(*self._response_len_range))
            pr = int(self._rng.choice([1, 2, 3], p=self._priority_weights))
            slack = int(self._rng.integers(*self._deadline_slack_range))
            deadline = current_time + p + o + slack

            requests.append(
                Request(
                    id=self._next_id,
                    prompt_tokens=p,
                    target_response_tokens=o,
                    priority=pr,
                    arrival_time=current_time,
                    deadline=deadline,
                )
            )
            self._next_id += 1
        return requests

    def is_exhausted(self, current_time: int) -> bool:
        """Return True when no future arrivals can be generated."""
        horizon_exhausted = (
            self._arrival_horizon is not None
            and current_time >= self._arrival_horizon
        )
        count_exhausted = (
            self._max_arrivals is not None
            and self._next_id >= self._max_arrivals
        )
        if self._arrival_horizon is None and self._max_arrivals is None:
            return False
        return horizon_exhausted or count_exhausted

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._next_id = 0
