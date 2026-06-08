"""LLMEnvSimple - token-based Gymnasium environment."""

from __future__ import annotations

from math import ceil

import gymnasium

from core.constants import (
    BATCH_MAX_SIZE,
    CLIENT_TIMEOUT_AFTER_DEADLINE,
    CPU_RAM_MAX_TOKENS,
    GPU_DECODE_TOKENS_PER_STEP,
    GPU_VRAM_MAX_TOKENS,
    MAX_CONCURRENT_PREFILL,
    MAX_EPISODE_STEPS,
    PREFILL_DECODE_SLOT_COST,
    PREFILL_TOKENS_PER_STEP,
    SWAP_IN_DELAY,
)
from core.types import (
    ActionType,
    PreemptStrategy,
    PreemptedLabel,
    Request,
    RequestStage,
    SchedulerAction,
    StepMetrics,
)
from core.workload import WorkloadGenerator
from env.memory import MemoryPool
from env.metrics import MetricsCollector
from env.reward import RewardCalculator, RewardWeights


class LLMEnvSimple(gymnasium.Env):
    """Token-based LLM serving simulator.

    The environment keeps the original high-level phases but avoids the most
    misleading shortcuts: prefill has prompt-dependent latency, decode has a
    per-step compute budget, timeout applies to all unfinished requests, and
    episode metrics include unfinished backlog.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        workload_generator: WorkloadGenerator,
        reward_weights: RewardWeights | None = None,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        super().__init__()
        self._wg = workload_generator
        self._memory = MemoryPool(GPU_VRAM_MAX_TOKENS, CPU_RAM_MAX_TOKENS)
        self._reward_calc = RewardCalculator(reward_weights)
        self._metrics = MetricsCollector()
        self._max_episode_steps = max_episode_steps

        self._active_batch: list[Request] = []
        self._queue: list[Request] = []
        self._preempted_queue: list[Request] = []
        self._current_time: int = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._memory.reset()
        self._metrics.reset()
        self._active_batch.clear()
        self._queue.clear()
        self._preempted_queue.clear()
        self._current_time = 0

        self._wg.reset(seed=seed)
        self._add_arrivals(self._current_time)

        return self.state_snapshot, {}

    def step(
        self, action: SchedulerAction
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute one scheduling step."""
        step_time = self._current_time

        p_recompute = 0.0
        p_oom = 0
        p_cpu_overflow = 0
        abort_priority_sum = 0.0

        phase1 = self._execute_action(action)
        p_recompute += phase1["recompute"]
        p_cpu_overflow += phase1["cpu_overflow"]

        phase2 = self._process_gpu()
        tokens_decoded = phase2["tokens_decoded"]
        p_oom += phase2["oom"]

        phase3 = self._post_process()
        abort_priority_sum += phase3["abort_priority_sum"]

        sla_priority_sum = self._check_sla_violations(step_time)

        wait_sum = self._queued_wait_sum(step_time)

        self._current_time += 1
        if not self._hit_time_limit():
            self._add_arrivals(self._current_time)

        terminated = self._is_episode_done()
        truncated = False
        crash_sum = self._unfinished_priority_sum() if self._hit_time_limit() else 0.0

        reward = self._reward_calc.compute(
            tokens_decoded=tokens_decoded,
            wait_sum=wait_sum,
            sla_sum=sla_priority_sum,
            abort_sum=abort_priority_sum,
            recompute_sum=p_recompute,
            oom_count=p_oom,
            cpu_overflow_count=p_cpu_overflow,
            crash_sum=crash_sum,
        )

        self._metrics.on_step(tokens_decoded)
        step_metrics = StepMetrics(
            tokens_decoded=tokens_decoded,
            requests_completed=phase3["completed"],
            requests_aborted=phase3["aborted"],
            sla_violations=int(sla_priority_sum > 0),
            oom_events=int(p_oom > 0),
            active_batch_size=len(self._active_batch),
            queue_size=len(self._queue),
            preempted_queue_size=len(self._preempted_queue),
            gpu_utilization=self._memory.gpu_used / GPU_VRAM_MAX_TOKENS,
            cpu_utilization=self._memory.cpu_used / CPU_RAM_MAX_TOKENS,
        )

        info = {
            "step_metrics": step_metrics,
            "episode_metrics": (
                self._metrics.get_summary(self._unfinished_count())
                if terminated
                else None
            ),
        }

        return self.state_snapshot, reward, terminated, truncated, info

    @property
    def state_snapshot(self) -> dict:
        """Read-only snapshot for observation encoder and schedulers."""
        return {
            "active_batch": list(self._active_batch),
            "queue": list(self._queue),
            "preempted_queue": list(self._preempted_queue),
            "gpu_tokens_used": self._memory.gpu_used,
            "cpu_tokens_used": self._memory.cpu_used,
            "gpu_tokens_max": GPU_VRAM_MAX_TOKENS,
            "cpu_tokens_max": CPU_RAM_MAX_TOKENS,
            "current_time": self._current_time,
        }

    def _add_arrivals(self, current_time: int) -> None:
        arrivals = self._wg.generate_arrivals(current_time)
        self._queue.extend(arrivals)
        self._metrics.on_requests_arrived(len(arrivals))

    def _execute_action(self, action: SchedulerAction) -> dict:
        penalties = {"recompute": 0.0, "cpu_overflow": 0}

        if action.action_type == ActionType.PROMOTE:
            self._execute_promote(action.indices)
        elif action.action_type == ActionType.PREEMPT:
            penalties = self._execute_preempt(action.indices, action.preempt_strategy)
        elif action.action_type == ActionType.RESUME:
            self._execute_resume(action.indices)

        return penalties

    def _execute_promote(self, indices: list[int]) -> None:
        prefills_this_step = self._count_prefills_this_step()
        promoted: list[tuple[int, Request]] = []

        for idx in indices:
            if idx < 0 or idx >= len(self._queue):
                continue
            if len(self._active_batch) >= BATCH_MAX_SIZE:
                continue
            if prefills_this_step >= MAX_CONCURRENT_PREFILL:
                continue

            req = self._queue[idx]
            if not self._memory.gpu_alloc(req.prompt_tokens):
                continue

            req.stage = RequestStage.PREFILL
            req.prefill_remaining = self._prefill_steps(req.prompt_tokens)
            req.tokens_generated = 0
            req.ttft = None
            promoted.append((idx, req))
            self._active_batch.append(req)
            prefills_this_step += 1

        for idx, _ in sorted(promoted, key=lambda item: item[0], reverse=True):
            self._queue.pop(idx)

    def _execute_preempt(
        self, indices: list[int], strategy: PreemptStrategy
    ) -> dict:
        penalties = {"recompute": 0.0, "cpu_overflow": 0}
        to_remove = []

        for idx in sorted(set(indices)):
            if idx < 0 or idx >= len(self._active_batch):
                continue
            req = self._active_batch[idx]
            if req.stage == RequestStage.PREFILL:
                continue

            if req.stage == RequestStage.SWAP_IN_DEGRADED:
                self._memory.gpu_free_tokens(req.total_tokens)
                if strategy == PreemptStrategy.RECOMPUTE:
                    self._memory.cpu_free_tokens(req.cpu_tokens_held)
                    req.cpu_tokens_held = 0
                    req.preempted_label = PreemptedLabel.RECOMPUTE_WAITING
                    penalties["recompute"] += float(req.tokens_generated)
                else:
                    req.preempted_label = PreemptedLabel.SWAPPED_TO_CPU
                req.stage = None
                req.swap_in_remaining = 0
                self._preempted_queue.append(req)
                to_remove.append(idx)

            elif req.stage == RequestStage.DECODE:
                if strategy == PreemptStrategy.RECOMPUTE:
                    self._memory.gpu_free_tokens(req.total_tokens)
                    penalties["recompute"] += float(req.tokens_generated)
                    req.tokens_generated = 0
                    req.prefill_remaining = 0
                    req.preempted_label = PreemptedLabel.RECOMPUTE_WAITING
                    req.stage = None
                    self._preempted_queue.append(req)
                    to_remove.append(idx)
                elif self._memory.cpu_alloc(req.total_tokens):
                    self._memory.gpu_free_tokens(req.total_tokens)
                    req.cpu_tokens_held = req.total_tokens
                    req.preempted_label = PreemptedLabel.SWAPPED_TO_CPU
                    req.stage = None
                    self._preempted_queue.append(req)
                    to_remove.append(idx)
                else:
                    penalties["cpu_overflow"] += 1

        for idx in sorted(to_remove, reverse=True):
            self._active_batch.pop(idx)

        return penalties

    def _execute_resume(self, indices: list[int]) -> None:
        prefills_this_step = self._count_prefills_this_step()
        to_remove = []

        for idx in indices:
            if idx < 0 or idx >= len(self._preempted_queue):
                continue
            if len(self._active_batch) >= BATCH_MAX_SIZE:
                continue

            req = self._preempted_queue[idx]
            if req.preempted_label == PreemptedLabel.RECOMPUTE_WAITING:
                if prefills_this_step >= MAX_CONCURRENT_PREFILL:
                    continue
                if not self._memory.gpu_alloc(req.prompt_tokens):
                    continue

                req.stage = RequestStage.PREFILL
                req.prefill_remaining = self._prefill_steps(req.prompt_tokens)
                req.tokens_generated = 0
                req.preempted_label = None
                self._active_batch.append(req)
                to_remove.append(idx)
                prefills_this_step += 1

            elif req.preempted_label == PreemptedLabel.SWAPPED_TO_CPU:
                if not self._memory.gpu_alloc(req.total_tokens):
                    continue

                req.stage = RequestStage.SWAP_IN_DEGRADED
                req.swap_in_remaining = SWAP_IN_DELAY
                req.preempted_label = None
                self._active_batch.append(req)
                to_remove.append(idx)

        for idx in sorted(to_remove, reverse=True):
            self._preempted_queue.pop(idx)

    def _process_gpu(self) -> dict:
        tokens_decoded = 0
        oom_count = 0
        prefill_slots_used = 0
        decode_next_step: set[int] = set()

        for req in self._active_batch:
            if req.stage != RequestStage.PREFILL:
                continue
            req.prefill_remaining -= 1
            prefill_slots_used += PREFILL_DECODE_SLOT_COST
            if req.prefill_remaining <= 0:
                req.prefill_remaining = 0
                req.ttft = self._current_time - req.arrival_time + 1
                req.stage = RequestStage.DECODE
                decode_next_step.add(req.id)

        for req in self._active_batch:
            if req.stage != RequestStage.SWAP_IN_DEGRADED:
                continue
            req.swap_in_remaining -= 1
            if req.swap_in_remaining <= 0:
                req.swap_in_remaining = 0
                req.stage = RequestStage.DECODE
                self._memory.cpu_free_tokens(req.cpu_tokens_held)
                req.cpu_tokens_held = 0
                decode_next_step.add(req.id)

        decode_budget = max(0, GPU_DECODE_TOKENS_PER_STEP - prefill_slots_used)
        for req in self._active_batch:
            if decode_budget <= 0:
                break
            if req.stage != RequestStage.DECODE:
                continue
            if req.id in decode_next_step:
                continue
            if self._memory.gpu_used + 1 > GPU_VRAM_MAX_TOKENS:
                oom_count += 1
                continue
            self._memory.gpu_used += 1
            req.tokens_generated += 1
            tokens_decoded += 1
            decode_budget -= 1

        return {"tokens_decoded": tokens_decoded, "oom": oom_count}

    def _post_process(self) -> dict:
        completed = 0
        aborted = 0
        abort_priority_sum = 0.0

        remaining_active = []
        for req in self._active_batch:
            if req.is_complete:
                self._free_request_memory(req)
                self._metrics.on_request_complete(req, self._current_time)
                completed += 1
            else:
                remaining_active.append(req)
        self._active_batch = remaining_active

        self._active_batch, active_aborted, active_priority = self._abort_timed_out(
            self._active_batch
        )
        self._queue, queue_aborted, queue_priority = self._abort_timed_out(self._queue)
        self._preempted_queue, preempted_aborted, preempted_priority = (
            self._abort_timed_out(self._preempted_queue)
        )

        aborted = active_aborted + queue_aborted + preempted_aborted
        abort_priority_sum = active_priority + queue_priority + preempted_priority

        return {
            "completed": completed,
            "aborted": aborted,
            "abort_priority_sum": abort_priority_sum,
        }

    def _abort_timed_out(self, requests: list[Request]) -> tuple[list[Request], int, float]:
        kept = []
        aborted = 0
        priority_sum = 0.0
        for req in requests:
            if self._current_time - req.deadline > CLIENT_TIMEOUT_AFTER_DEADLINE:
                self._free_request_memory(req)
                self._metrics.on_request_abort(req)
                aborted += 1
                priority_sum += req.priority
            else:
                kept.append(req)
        return kept, aborted, priority_sum

    def _free_request_memory(self, req: Request) -> None:
        if req.stage in (
            RequestStage.PREFILL,
            RequestStage.SWAP_IN_DEGRADED,
            RequestStage.DECODE,
        ):
            self._memory.gpu_free_tokens(req.total_tokens)
        if req.cpu_tokens_held > 0:
            self._memory.cpu_free_tokens(req.cpu_tokens_held)
            req.cpu_tokens_held = 0

    def _check_sla_violations(self, step_time: int) -> float:
        sla_sum = 0.0
        for req in self._unfinished_requests():
            if step_time == req.deadline + 1:
                sla_sum += req.priority
                self._metrics.on_sla_violation(req)
        return sla_sum

    def _queued_wait_sum(self, step_time: int) -> float:
        return float(
            sum(step_time - r.arrival_time for r in self._queue)
            + sum(step_time - r.arrival_time for r in self._preempted_queue)
        )

    def _count_prefills_this_step(self) -> int:
        return sum(1 for r in self._active_batch if r.stage == RequestStage.PREFILL)

    def _prefill_steps(self, prompt_tokens: int) -> int:
        return max(1, ceil(prompt_tokens / PREFILL_TOKENS_PER_STEP))

    def _unfinished_requests(self) -> list[Request]:
        return self._active_batch + self._queue + self._preempted_queue

    def _unfinished_count(self) -> int:
        return len(self._active_batch) + len(self._queue) + len(self._preempted_queue)

    def _unfinished_priority_sum(self) -> float:
        return float(sum(req.priority for req in self._unfinished_requests()))

    def _hit_time_limit(self) -> bool:
        return self._current_time >= self._max_episode_steps

    def _is_episode_done(self) -> bool:
        if self._hit_time_limit():
            return True
        if self._unfinished_count() > 0:
            return False
        return self._wg.is_exhausted(self._current_time)
