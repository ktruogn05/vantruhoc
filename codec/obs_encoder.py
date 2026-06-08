"""Observation encoder - state snapshot to fixed-size numpy tensor."""

import gymnasium
import numpy as np

from core.constants import (
    BATCH_MAX_SIZE,
    CPU_RAM_MAX_TOKENS,
    GPU_VRAM_MAX_TOKENS,
    MAX_EPISODE_STEPS,
    MAX_RESPONSE_TOKENS,
    PREFILL_TOKENS_PER_STEP,
    SWAP_IN_DELAY,
)
from core.types import PreemptedLabel, Request, RequestStage

K_QUEUE = 32
K_PREEMPTED = 16
F_GLOBAL = 7
F_BATCH = 14
F_QUEUE = 7
F_PREEMPTED = 11

OBS_DIM = (
    F_GLOBAL
    + BATCH_MAX_SIZE * F_BATCH
    + K_QUEUE * F_QUEUE
    + K_PREEMPTED * F_PREEMPTED
)

P_MAX = 2048.0
O_MAX = float(MAX_RESPONSE_TOKENS)
T_MAX = float(MAX_EPISODE_STEPS)
C_MAX = float(CPU_RAM_MAX_TOKENS)
D_MAX = 500.0
PREFILL_MAX_STEPS = max(1.0, P_MAX / PREFILL_TOKENS_PER_STEP)


class ObservationEncoder:
    """Encode variable-length env state to a fixed-size float32 array."""

    OBS_DIM = OBS_DIM

    def __init__(self):
        self._obs_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return self._obs_space

    def encode(self, state: dict) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        pos = 0

        queue_len = len(state["queue"])
        preempted_len = len(state["preempted_queue"])
        active_len = len(state["active_batch"])

        obs[pos] = _ratio(state["gpu_tokens_used"], GPU_VRAM_MAX_TOKENS)
        obs[pos + 1] = _ratio(state["cpu_tokens_used"], CPU_RAM_MAX_TOKENS)
        obs[pos + 2] = _clamp01(1.0 - state["current_time"] / T_MAX)
        obs[pos + 3] = _ratio(queue_len, K_QUEUE)
        obs[pos + 4] = _ratio(preempted_len, K_PREEMPTED)
        obs[pos + 5] = _ratio(active_len, BATCH_MAX_SIZE)
        obs[pos + 6] = _ratio(max(queue_len - K_QUEUE, 0), K_QUEUE)
        pos += F_GLOBAL

        for i in range(BATCH_MAX_SIZE):
            if i < active_len:
                req: Request = state["active_batch"][i]
                pos = _encode_active(obs, pos, req, state["current_time"])
            else:
                pos += F_BATCH

        queue = state["queue"]
        for i in range(K_QUEUE):
            if i < queue_len:
                pos = _encode_queue(obs, pos, queue[i], state["current_time"])
            else:
                pos += F_QUEUE

        preempted = state["preempted_queue"]
        for i in range(K_PREEMPTED):
            if i < preempted_len:
                pos = _encode_preempted(obs, pos, preempted[i], state["current_time"])
            else:
                pos += F_PREEMPTED

        return obs


def _encode_active(obs: np.ndarray, pos: int, req: Request, now: int) -> int:
    obs[pos] = 1.0
    if req.stage is not None:
        obs[pos + 1 + int(req.stage)] = 1.0
    obs[pos + 4] = _ratio(req.prompt_tokens, P_MAX)
    obs[pos + 5] = _ratio(req.target_response_tokens, O_MAX)
    obs[pos + 6] = _ratio(req.tokens_generated, O_MAX)
    obs[pos + 7] = _progress(req)
    obs[pos + 8] = req.priority / 3.0
    obs[pos + 9] = _ratio(now - req.arrival_time, T_MAX)
    obs[pos + 10] = _deadline_urgency(req, now)
    obs[pos + 11] = _ratio(req.swap_in_remaining, SWAP_IN_DELAY)
    obs[pos + 12] = _ratio(req.prefill_remaining, PREFILL_MAX_STEPS)
    obs[pos + 13] = _ratio(req.total_tokens, GPU_VRAM_MAX_TOKENS)
    return pos + F_BATCH


def _encode_queue(obs: np.ndarray, pos: int, req: Request, now: int) -> int:
    obs[pos] = 1.0
    obs[pos + 1] = _ratio(req.prompt_tokens, P_MAX)
    obs[pos + 2] = _ratio(req.target_response_tokens, O_MAX)
    obs[pos + 3] = req.priority / 3.0
    obs[pos + 4] = _ratio(now - req.arrival_time, T_MAX)
    obs[pos + 5] = _deadline_urgency(req, now)
    obs[pos + 6] = _ratio(req.prompt_tokens + req.target_response_tokens, P_MAX + O_MAX)
    return pos + F_QUEUE


def _encode_preempted(obs: np.ndarray, pos: int, req: Request, now: int) -> int:
    obs[pos] = 1.0
    if req.preempted_label is not None:
        obs[pos + 1 + int(req.preempted_label)] = 1.0
    obs[pos + 3] = _ratio(req.prompt_tokens, P_MAX)
    obs[pos + 4] = _ratio(req.target_response_tokens, O_MAX)
    obs[pos + 5] = _ratio(req.tokens_generated, O_MAX)
    obs[pos + 6] = _progress(req)
    obs[pos + 7] = req.priority / 3.0
    obs[pos + 8] = _ratio(now - req.arrival_time, T_MAX)
    obs[pos + 9] = _deadline_urgency(req, now)
    obs[pos + 10] = _ratio(req.cpu_tokens_held, C_MAX)
    return pos + F_PREEMPTED


def _progress(req: Request) -> float:
    return _ratio(req.tokens_generated, max(req.target_response_tokens, 1))


def _deadline_urgency(req: Request, now: int) -> float:
    time_to_deadline = req.deadline - now
    return 1.0 - _clamp01(time_to_deadline / D_MAX)


def _ratio(value: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return _clamp01(value / denom)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
