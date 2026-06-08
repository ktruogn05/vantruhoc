import unittest

from codec.action_decoder import ActionDecoder
from codec.obs_encoder import F_BATCH, F_GLOBAL, F_QUEUE, K_QUEUE, ObservationEncoder
from configs.default import WorkloadConfig
from core.constants import (
    BATCH_MAX_SIZE,
    GPU_DECODE_TOKENS_PER_STEP,
    GPU_VRAM_MAX_TOKENS,
)
from core.types import (
    ActionType,
    PreemptStrategy,
    PreemptedLabel,
    Request,
    RequestStage,
    SchedulerAction,
)
from core.workload import WorkloadGenerator
from env.llm_env import LLMEnvSimple
from env.reward import RewardWeights
from scheduler.fcfs import FCFSScheduler
from training.env_factory import build_workload


class FixedWorkload:
    def __init__(self, arrivals_by_time):
        self.arrivals_by_time = arrivals_by_time
        self.reset_calls = []

    def reset(self, seed=None):
        self.reset_calls.append(seed)

    def generate_arrivals(self, current_time):
        return [req for req in self.arrivals_by_time.get(current_time, [])]

    def is_exhausted(self, current_time):
        return current_time >= max(self.arrivals_by_time.keys(), default=0)


def req(
    request_id=0,
    prompt=64,
    target=16,
    priority=1,
    arrival=0,
    deadline=10_000,
):
    return Request(
        id=request_id,
        prompt_tokens=prompt,
        target_response_tokens=target,
        priority=priority,
        arrival_time=arrival,
        deadline=deadline,
    )


class PipelineRegressionTests(unittest.TestCase):
    def test_workload_reset_none_does_not_replay_same_episode(self):
        wg = WorkloadGenerator(arrival_rate=5.0, seed=7, max_arrivals=5)
        wg.set_distributions(prompt_len=(64, 512), response_len=(32, 128))

        wg.reset(seed=7)
        first = [(r.prompt_tokens, r.target_response_tokens) for r in wg.generate_arrivals(0)]
        wg.reset(seed=7)
        repeated = [(r.prompt_tokens, r.target_response_tokens) for r in wg.generate_arrivals(0)]
        wg.reset(seed=None)
        next_episode = [
            (r.prompt_tokens, r.target_response_tokens) for r in wg.generate_arrivals(0)
        ]

        self.assertEqual(first, repeated)
        self.assertNotEqual(repeated, next_episode)

    def test_abort_reward_uses_priority_sum_once(self):
        late = req(priority=2, deadline=-31)
        env = LLMEnvSimple(
            FixedWorkload({0: [late]}),
            RewardWeights(
                throughput=0,
                wait=0,
                sla=0,
                abort=20,
                recompute=0,
                oom=0,
                cpu_overflow=0,
                crash=0,
            ),
        )
        env.reset(seed=1)
        _, reward, _, _, info = env.step(SchedulerAction(ActionType.NOOP))

        self.assertEqual(reward, -40)
        self.assertEqual(info["step_metrics"].requests_aborted, 1)

    def test_episode_can_end_after_finite_workload_drains(self):
        env = LLMEnvSimple(FixedWorkload({}), max_episode_steps=100)
        env.reset(seed=1)
        _, _, terminated, _, info = env.step(SchedulerAction(ActionType.NOOP))

        self.assertTrue(terminated)
        self.assertEqual(info["episode_metrics"]["total_requests"], 0)

    def test_terminal_metrics_count_unfinished_backlog(self):
        env = LLMEnvSimple(FixedWorkload({0: [req()]}), max_episode_steps=1)
        env.reset(seed=1)
        _, _, terminated, _, info = env.step(SchedulerAction(ActionType.NOOP))

        self.assertTrue(terminated)
        metrics = info["episode_metrics"]
        self.assertEqual(metrics["requests_arrived"], 1)
        self.assertEqual(metrics["requests_unfinished"], 1)
        self.assertEqual(metrics["total_requests"], 1)
        self.assertEqual(metrics["completion_rate"], 0)

    def test_observation_exposes_queue_target_response_length(self):
        encoder = ObservationEncoder()
        state = {
            "active_batch": [],
            "queue": [req(target=400)],
            "preempted_queue": [],
            "gpu_tokens_used": 0,
            "cpu_tokens_used": 0,
            "current_time": 0,
        }
        obs = encoder.encode(state)
        first_queue_pos = F_GLOBAL + BATCH_MAX_SIZE * F_BATCH

        self.assertEqual(K_QUEUE, 32)
        self.assertAlmostEqual(obs[first_queue_pos + 2], 400 / 512)
        self.assertEqual(len(obs), encoder.observation_space.shape[0])

    def test_action_space_includes_multi_preempt(self):
        decoder = ActionDecoder()
        batch = [req(0), req(1)]
        for r in batch:
            r.stage = RequestStage.DECODE
        state = {
            "active_batch": batch,
            "queue": [],
            "preempted_queue": [],
            "gpu_tokens_used": 128,
            "cpu_tokens_used": 0,
        }
        mask = decoder.get_action_mask(state)

        found = False
        for aid in range(decoder.N_ACTIONS):
            action = decoder.decode(aid)
            if (
                action.action_type == ActionType.PREEMPT
                and action.indices == [0, 1]
                and action.preempt_strategy == PreemptStrategy.SWAP
            ):
                found = True
                self.assertTrue(mask[aid])
                break
        self.assertTrue(found)

    def test_baseline_resumes_preempted_requests(self):
        paused = req(prompt=64)
        paused.preempted_label = PreemptedLabel.RECOMPUTE_WAITING
        state = {
            "active_batch": [],
            "queue": [req(1)],
            "preempted_queue": [paused],
            "gpu_tokens_used": 0,
        }

        action = FCFSScheduler().select_action(state)
        self.assertEqual(action.action_type, ActionType.RESUME)
        self.assertEqual(action.indices, [0])

    def test_prefill_latency_depends_on_prompt_and_decodes_next_step(self):
        long_prompt = req(prompt=512, target=1)
        env = LLMEnvSimple(FixedWorkload({0: [long_prompt]}), max_episode_steps=20)
        env.reset(seed=1)

        env.step(SchedulerAction(ActionType.PROMOTE, indices=[0]))
        active = env.state_snapshot["active_batch"][0]
        self.assertEqual(active.stage, RequestStage.PREFILL)
        self.assertEqual(active.prefill_remaining, 1)
        self.assertEqual(active.tokens_generated, 0)

        env.step(SchedulerAction(ActionType.NOOP))
        active = env.state_snapshot["active_batch"][0]
        self.assertEqual(active.stage, RequestStage.DECODE)
        self.assertEqual(active.ttft, 2)
        self.assertEqual(active.tokens_generated, 0)

        _, _, terminated, _, info = env.step(SchedulerAction(ActionType.NOOP))
        self.assertTrue(terminated)
        self.assertEqual(info["step_metrics"].requests_completed, 1)

    def test_prefill_work_reduces_decode_budget(self):
        env = LLMEnvSimple(FixedWorkload({}), max_episode_steps=10)
        env.reset(seed=1)
        env._active_batch = []
        env._memory.gpu_used = 0

        for i in range(BATCH_MAX_SIZE):
            decode_req = req(i, prompt=1, target=2)
            decode_req.stage = RequestStage.DECODE
            env._active_batch.append(decode_req)
            env._memory.gpu_used += decode_req.total_tokens

        for i in range(2):
            prefill_req = req(100 + i, prompt=512, target=1)
            prefill_req.stage = RequestStage.PREFILL
            prefill_req.prefill_remaining = 2
            env._active_batch.append(prefill_req)
            env._memory.gpu_used += prefill_req.total_tokens

        phase = env._process_gpu()
        self.assertEqual(phase["tokens_decoded"], GPU_DECODE_TOKENS_PER_STEP - 8)

    def test_env_factory_applies_distribution_config(self):
        config = WorkloadConfig(
            arrival_rate=5.0,
            seed=1,
            arrival_horizon=0,
            max_arrivals=1,
            prompt_len=(10, 11),
            response_len=(20, 21),
            deadline_slack=(30, 31),
        )
        workload = build_workload(config)
        arrivals = workload.generate_arrivals(0)

        self.assertEqual(len(arrivals), 1)
        self.assertEqual(arrivals[0].prompt_tokens, 10)
        self.assertEqual(arrivals[0].target_response_tokens, 20)


if __name__ == "__main__":
    unittest.main()
