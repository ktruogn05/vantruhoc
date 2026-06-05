"""LLMEnvSimple — 4-phase Gymnasium environment implementing env_simple.md spec."""

from __future__ import annotations

import gymnasium

from core.constants import (
    BATCH_MAX_SIZE,
    CLIENT_TIMEOUT_AFTER_DEADLINE,
    GPU_VRAM_MAX_TOKENS,
    CPU_RAM_MAX_TOKENS,
    MAX_CONCURRENT_PREFILL,
    MAX_EPISODE_STEPS,
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

    Implements the exact 4-phase step loop from env_simple.md.
    Works with SchedulerAction structs — encoding/decoding is external.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        workload_generator: WorkloadGenerator,
        reward_weights: RewardWeights | None = None,
    ):
        super().__init__()
        self._wg = workload_generator
        self._memory = MemoryPool(GPU_VRAM_MAX_TOKENS, CPU_RAM_MAX_TOKENS)
        self._reward_calc = RewardCalculator(reward_weights)
        self._metrics = MetricsCollector()

        # State
        self._active_batch: list[Request] = []
        self._queue: list[Request] = []
        self._preempted_queue: list[Request] = []
        self._current_time: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

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

        reset_seed = seed if seed is not None else 42
        self._wg.reset(seed=reset_seed)

        # Generate initial arrivals at t=0
        arrivals = self._wg.generate_arrivals(self._current_time)
        self._queue.extend(arrivals)

        return self.state_snapshot, {}

    def step(
        self, action: SchedulerAction
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute 4 phases per env_simple.md."""

        # Accumulators for this step
        p_abort = 0.0
        p_recompute = 0.0
        p_oom = 0.0
        p_cpu_overflow = 0.0

        # ── Phase 1: Execute scheduler action ──
        phase1 = self._execute_action(action)
        p_abort += phase1["abort"]
        p_recompute += phase1["recompute"]
        p_oom += phase1["oom"]
        p_cpu_overflow += phase1["cpu_overflow"]

        # ── Phase 2: GPU processing ──
        phase2 = self._process_gpu()
        tokens_decoded = phase2["tokens_decoded"]
        p_oom += phase2["oom"]

        # ── Phase 3: Completion, timeout, time advance ──
        phase3 = self._post_process()
        p_abort += phase3["abort"]

        # SLA check: fires at t == d_k + 1
        sla_sum = self._check_sla_violations()

        # Time advance
        self._current_time += 1

        # New arrivals
        arrivals = self._wg.generate_arrivals(self._current_time)
        self._queue.extend(arrivals)

        # Episode termination
        terminated = self._is_episode_done()
        truncated = False

        # ── Phase 4: Reward ──
        # Wait penalty: sum of (t - a_i) for all requests in queues
        wait_sum = 0.0
        for r in self._queue:
            wait_sum += self._current_time - r.arrival_time
        for r in self._preempted_queue:
            wait_sum += self._current_time - r.arrival_time

        # Crash penalty at terminal step
        crash_sum = 0.0
        if terminated and self._current_time >= MAX_EPISODE_STEPS:
            for r in self._active_batch:
                crash_sum += r.priority
            for r in self._queue:
                crash_sum += r.priority
            for r in self._preempted_queue:
                crash_sum += r.priority

        reward = self._reward_calc.compute(
            tokens_decoded=tokens_decoded,
            wait_sum=wait_sum,
            sla_sum=sla_sum,
            abort_sum=p_abort,
            recompute_sum=p_recompute,
            oom_count=int(p_oom),
            cpu_overflow_count=int(p_cpu_overflow / 50.0) if p_cpu_overflow > 0 else 0,
            crash_sum=crash_sum,
        )

        # Metrics
        self._metrics.on_step(tokens_decoded)
        step_metrics = StepMetrics(
            tokens_decoded=tokens_decoded,
            requests_completed=phase3["completed"],
            requests_aborted=phase3["aborted"],
            sla_violations=int(sla_sum > 0),
            oom_events=int(p_oom > 0),
            active_batch_size=len(self._active_batch),
            queue_size=len(self._queue),
            preempted_queue_size=len(self._preempted_queue),
            gpu_utilization=self._memory.gpu_used / GPU_VRAM_MAX_TOKENS,
            cpu_utilization=self._memory.cpu_used / CPU_RAM_MAX_TOKENS,
        )

        info = {
            "step_metrics": step_metrics,
            "episode_metrics": self._metrics.get_summary() if terminated else None,
        }

        return self.state_snapshot, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    @property
    def state_snapshot(self) -> dict:
        """Read-only snapshot for observation encoder / baseline schedulers."""
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

    # ------------------------------------------------------------------
    # Phase 1: Execute scheduler action
    # ------------------------------------------------------------------

    def _execute_action(self, action: SchedulerAction) -> dict:
        penalties = {"abort": 0.0, "recompute": 0.0, "oom": 0.0, "cpu_overflow": 0.0}

        if action.action_type == ActionType.PROMOTE:
            self._execute_promote(action.indices)
        elif action.action_type == ActionType.PREEMPT:
            p = self._execute_preempt(action.indices, action.preempt_strategy)
            penalties["recompute"] += p["recompute"]
            penalties["cpu_overflow"] += p["cpu_overflow"]
        elif action.action_type == ActionType.RESUME:
            self._execute_resume(action.indices)
        # NOOP: do nothing

        return penalties

    def _execute_promote(self, indices: list[int]) -> None:
        """Promote requests from queue to active batch."""
        prefills_this_step = self._count_prefills_this_step()

        # Sort descending to avoid index shifting on removal
        for idx in sorted(indices, reverse=True):
            if idx < 0 or idx >= len(self._queue):
                continue
            if len(self._active_batch) >= BATCH_MAX_SIZE:
                continue
            if prefills_this_step >= MAX_CONCURRENT_PREFILL:
                continue

            req = self._queue[idx]
            if not self._memory.gpu_alloc(req.prompt_tokens):
                continue

            # Success — move to active batch
            req.stage = RequestStage.PREFILL
            req.tokens_generated = 0
            self._active_batch.append(req)
            self._queue.pop(idx)
            prefills_this_step += 1

    def _execute_preempt(
        self, indices: list[int], strategy: PreemptStrategy
    ) -> dict:
        """Preempt requests from active batch."""
        penalties = {"recompute": 0.0, "cpu_overflow": 0.0}

        # Collect valid requests first, then remove (avoid index shifting)
        to_remove = []
        for idx in indices:
            if idx < 0 or idx >= len(self._active_batch):
                continue
            req = self._active_batch[idx]

            # Cannot preempt PREFILL (atomic)
            if req.stage == RequestStage.PREFILL:
                continue

            if req.stage == RequestStage.SWAP_IN_DEGRADED:
                # Free GPU VRAM
                self._memory.gpu_free_tokens(req.total_tokens)

                if strategy == PreemptStrategy.RECOMPUTE:
                    # Free CPU RAM
                    self._memory.cpu_free_tokens(req.cpu_tokens_held)
                    req.cpu_tokens_held = 0
                    req.preempted_label = PreemptedLabel.RECOMPUTE_WAITING
                    penalties["recompute"] += 1.0 * req.tokens_generated
                else:  # SWAP
                    # CPU RAM stays, label as swapped
                    req.preempted_label = PreemptedLabel.SWAPPED_TO_CPU
                    # cpu_tokens_held remains

                req.stage = None
                req.swap_in_remaining = 0
                self._preempted_queue.append(req)
                to_remove.append(idx)

            elif req.stage == RequestStage.DECODE:
                if strategy == PreemptStrategy.RECOMPUTE:
                    self._memory.gpu_free_tokens(req.total_tokens)
                    penalties["recompute"] += 1.0 * req.tokens_generated
                    req.tokens_generated = 0
                    req.preempted_label = PreemptedLabel.RECOMPUTE_WAITING
                    req.stage = None
                    self._preempted_queue.append(req)
                    to_remove.append(idx)
                else:  # SWAP
                    if not self._memory.cpu_alloc(req.total_tokens):
                        # Swap fails — request stays in batch
                        penalties["cpu_overflow"] += 50.0
                    else:
                        self._memory.gpu_free_tokens(req.total_tokens)
                        req.cpu_tokens_held = req.total_tokens
                        req.preempted_label = PreemptedLabel.SWAPPED_TO_CPU
                        req.stage = None
                        self._preempted_queue.append(req)
                        to_remove.append(idx)

        # Remove from active batch (descending order)
        for idx in sorted(to_remove, reverse=True):
            self._active_batch.pop(idx)

        return penalties

    def _execute_resume(self, indices: list[int]) -> None:
        """Resume requests from preempted queue."""
        prefills_this_step = self._count_prefills_this_step()
        to_remove = []

        for idx in sorted(indices, reverse=True):
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
                # CPU RAM stays until swap-in completes

        for idx in sorted(to_remove, reverse=True):
            self._preempted_queue.pop(idx)

    def _count_prefills_this_step(self) -> int:
        """Count requests currently in PREFILL stage (just promoted this step)."""
        return sum(
            1 for r in self._active_batch if r.stage == RequestStage.PREFILL
        )

    # ------------------------------------------------------------------
    # Phase 2: GPU processing
    # ------------------------------------------------------------------

    def _process_gpu(self) -> dict:
        tokens_decoded = 0
        oom_count = 0

        for req in self._active_batch:
            if req.stage == RequestStage.PREFILL:
                # Process prompt. No token generated this step.
                req.ttft = self._current_time - req.arrival_time + 1
                req.stage = RequestStage.DECODE

            elif req.stage == RequestStage.SWAP_IN_DEGRADED:
                req.swap_in_remaining -= 1
                if req.swap_in_remaining == 0:
                    req.stage = RequestStage.DECODE
                    # Release CPU RAM
                    self._memory.cpu_free_tokens(req.cpu_tokens_held)
                    req.cpu_tokens_held = 0

            elif req.stage == RequestStage.DECODE:
                # Try to generate 1 token
                if self._memory.gpu_used + 1 > GPU_VRAM_MAX_TOKENS:
                    # OOM — skip token generation this step
                    oom_count += 1
                else:
                    self._memory.gpu_used += 1
                    req.tokens_generated += 1
                    tokens_decoded += 1

        return {"tokens_decoded": tokens_decoded, "oom": oom_count}

    # ------------------------------------------------------------------
    # Phase 3: Completion, timeout, time advance
    # ------------------------------------------------------------------

    def _post_process(self) -> dict:
        completed = 0
        aborted = 0
        abort_penalty = 0.0

        # Check completions
        remaining = []
        for req in self._active_batch:
            if req.is_complete:
                self._memory.gpu_free_tokens(req.total_tokens)
                self._metrics.on_request_complete(req, self._current_time)
                completed += 1
            else:
                remaining.append(req)
        self._active_batch = remaining

        # Client timeout — scan queue and preempted_queue
        def timeout_filter(req_list: list[Request]) -> list[Request]:
            nonlocal aborted, abort_penalty
            kept = []
            for req in req_list:
                if self._current_time - req.deadline > CLIENT_TIMEOUT_AFTER_DEADLINE:
                    # Abort
                    if req.preempted_label == PreemptedLabel.SWAPPED_TO_CPU:
                        self._memory.cpu_free_tokens(req.cpu_tokens_held)
                    abort_penalty += 20.0 * req.priority
                    self._metrics.on_request_abort(req)
                    aborted += 1
                else:
                    kept.append(req)
            return kept

        self._queue = timeout_filter(self._queue)
        self._preempted_queue = timeout_filter(self._preempted_queue)

        return {
            "completed": completed,
            "aborted": aborted,
            "abort": abort_penalty,
        }

    def _check_sla_violations(self) -> float:
        """P_SLA: fires at t == d_k + 1 for all requests everywhere."""
        sla_sum = 0.0
        all_requests = (
            list(self._active_batch)
            + list(self._queue)
            + list(self._preempted_queue)
        )
        for req in all_requests:
            if self._current_time == req.deadline + 1:
                sla_sum += req.priority
                self._metrics.on_sla_violation(req)
        return sla_sum

    def _is_episode_done(self) -> bool:
        return self._current_time >= MAX_EPISODE_STEPS

