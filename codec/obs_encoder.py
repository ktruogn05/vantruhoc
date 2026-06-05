"""Observation encoder — state snapshot → fixed-size numpy tensor."""

import numpy as np
import gymnasium

from core.constants import (
    BATCH_MAX_SIZE,
    GPU_VRAM_MAX_TOKENS,
    CPU_RAM_MAX_TOKENS,
    MAX_EPISODE_STEPS,
    MAX_RESPONSE_TOKENS,
    SWAP_IN_DELAY,
)
from core.types import Request, RequestStage, PreemptedLabel

# Layout constants
K_QUEUE = 16        # visible queue positions
K_PREEMPTED = 8     # visible preempted positions
F_GLOBAL = 4
F_BATCH = 11        # per active batch slot
F_QUEUE = 5         # per queue slot
F_PREEMPTED = 8     # per preempted slot

OBS_DIM = F_GLOBAL + BATCH_MAX_SIZE * F_BATCH + K_QUEUE * F_QUEUE + K_PREEMPTED * F_PREEMPTED
# = 4 + 16*11 + 16*5 + 8*8 = 4 + 176 + 80 + 64 = 324

# Normalization constants
P_MAX = 2048.0      # generous upper bound for prompt length
O_MAX = float(MAX_RESPONSE_TOKENS)
T_MAX = float(MAX_EPISODE_STEPS)
C_MAX = float(CPU_RAM_MAX_TOKENS)


class ObservationEncoder:
    """Encode variable-length env state → fixed-size float32 array in [0, 1]."""

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

        # ── Global features (4) ──
        obs[pos] = state["gpu_tokens_used"] / GPU_VRAM_MAX_TOKENS
        obs[pos + 1] = state["cpu_tokens_used"] / CPU_RAM_MAX_TOKENS
        obs[pos + 2] = 1.0 - state["current_time"] / T_MAX  # time remaining ratio
        obs[pos + 3] = min(len(state["queue"]) / BATCH_MAX_SIZE, 1.0)  # queue pressure
        pos += F_GLOBAL

        # ── Active batch slots (16 × 11) ──
        for i in range(BATCH_MAX_SIZE):
            if i < len(state["active_batch"]):
                req: Request = state["active_batch"][i]
                obs[pos] = 1.0  # occupied
                # stage one-hot (3)
                if req.stage is not None:
                    obs[pos + 1 + int(req.stage)] = 1.0
                obs[pos + 4] = min(req.prompt_tokens / P_MAX, 1.0)
                obs[pos + 5] = min(req.tokens_generated / O_MAX, 1.0)
                # progress ratio
                obs[pos + 6] = (
                    req.tokens_generated / max(req.target_response_tokens, 1)
                )
                obs[pos + 7] = req.priority / 3.0
                obs[pos + 8] = min(
                    (state["current_time"] - req.arrival_time) / T_MAX, 1.0
                )
                # deadline urgency: how close to deadline (0 = far, 1 = past)
                time_to_deadline = req.deadline - state["current_time"]
                obs[pos + 9] = 1.0 - max(min(time_to_deadline / 500.0, 1.0), 0.0)
                obs[pos + 10] = req.swap_in_remaining / SWAP_IN_DELAY if SWAP_IN_DELAY > 0 else 0.0
            pos += F_BATCH

        # ── Queue slots (16 × 5) ──
        queue = state["queue"]
        for i in range(K_QUEUE):
            if i < len(queue):
                req = queue[i]
                obs[pos] = 1.0  # occupied
                obs[pos + 1] = min(req.prompt_tokens / P_MAX, 1.0)
                obs[pos + 2] = req.priority / 3.0
                obs[pos + 3] = min(
                    (state["current_time"] - req.arrival_time) / T_MAX, 1.0
                )
                time_to_deadline = req.deadline - state["current_time"]
                obs[pos + 4] = 1.0 - max(min(time_to_deadline / 500.0, 1.0), 0.0)
            pos += F_QUEUE

        # ── Preempted queue slots (8 × 8) ──
        preempted = state["preempted_queue"]
        for i in range(K_PREEMPTED):
            if i < len(preempted):
                req = preempted[i]
                obs[pos] = 1.0  # occupied
                # label one-hot (2)
                if req.preempted_label is not None:
                    obs[pos + 1 + int(req.preempted_label)] = 1.0
                obs[pos + 3] = min(req.prompt_tokens / P_MAX, 1.0)
                obs[pos + 4] = min(req.tokens_generated / O_MAX, 1.0)
                obs[pos + 5] = req.priority / 3.0
                obs[pos + 6] = min(
                    (state["current_time"] - req.arrival_time) / T_MAX, 1.0
                )
                obs[pos + 7] = min(req.cpu_tokens_held / C_MAX, 1.0)
            pos += F_PREEMPTED

        return obs
