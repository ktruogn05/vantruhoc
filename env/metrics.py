"""Episode-level metrics accumulator for evaluation."""

from core.types import Request


class MetricsCollector:
    """Collect episode-level stats for logging and scheduler comparison."""

    def __init__(self):
        self.reset()

    def on_requests_arrived(self, count: int) -> None:
        self._arrived += count

    def on_request_complete(self, request: Request, current_time: int) -> None:
        turnaround = current_time - request.arrival_time + 1
        self._turnarounds.append(turnaround)
        if request.ttft is not None:
            self._ttfts.append(request.ttft)
        self._completed += 1
        self._total_tokens_generated += request.tokens_generated

    def on_request_abort(self, request: Request) -> None:
        self._aborted += 1

    def on_sla_violation(self, request: Request) -> None:
        self._sla_violations += 1

    def on_step(self, tokens_decoded: int) -> None:
        self._total_steps += 1
        self._total_tokens_decoded += tokens_decoded

    def get_summary(self, unfinished_count: int = 0) -> dict:
        """Return episode summary dict."""
        n_ttft = len(self._ttfts)
        n_turn = len(self._turnarounds)
        total_requests = self._completed + self._aborted + unfinished_count
        total_requests = max(total_requests, self._arrived)

        return {
            "total_steps": self._total_steps,
            "requests_arrived": self._arrived,
            "requests_completed": self._completed,
            "requests_aborted": self._aborted,
            "requests_unfinished": unfinished_count,
            "total_requests": total_requests,
            "completion_rate": self._completed / max(total_requests, 1),
            "abort_rate": self._aborted / max(total_requests, 1),
            "unfinished_rate": unfinished_count / max(total_requests, 1),
            "sla_violations": self._sla_violations,
            "avg_ttft": sum(self._ttfts) / max(n_ttft, 1),
            "avg_turnaround": sum(self._turnarounds) / max(n_turn, 1),
            "total_tokens_decoded": self._total_tokens_decoded,
            "throughput_tokens_per_step": (
                self._total_tokens_decoded / max(self._total_steps, 1)
            ),
        }

    def reset(self) -> None:
        self._arrived: int = 0
        self._ttfts: list[int] = []
        self._turnarounds: list[int] = []
        self._completed: int = 0
        self._aborted: int = 0
        self._sla_violations: int = 0
        self._total_steps: int = 0
        self._total_tokens_decoded: int = 0
        self._total_tokens_generated: int = 0
