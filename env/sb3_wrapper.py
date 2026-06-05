"""SB3-compatible wrapper: LLMEnvSimple + Codec → numpy tensors."""

from __future__ import annotations

import numpy as np
import gymnasium

from core.types import SchedulerAction
from env.llm_env import LLMEnvSimple
from codec.obs_encoder import ObservationEncoder
from codec.action_decoder import ActionDecoder


class SB3EnvWrapper(gymnasium.Env):
    """Thin wrapper bridging LLMEnvSimple to SB3's MaskablePPO.

    - observation_space: Box from encoder
    - action_space: Discrete from decoder
    - step(): int → SchedulerAction → env.step() → obs tensor
    - action_masks(): for MaskablePPO
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: LLMEnvSimple):
        super().__init__()
        self._env = env
        self._encoder = ObservationEncoder()
        self._decoder = ActionDecoder()
        self.observation_space = self._encoder.observation_space
        self.action_space = self._decoder.action_space

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        state, info = self._env.reset(seed=seed, options=options)
        obs = self._encoder.encode(self._env.state_snapshot)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        scheduler_action: SchedulerAction = self._decoder.decode(action)
        _, reward, terminated, truncated, info = self._env.step(scheduler_action)
        obs = self._encoder.encode(self._env.state_snapshot)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO automatically each step."""
        return self._decoder.get_action_mask(self._env.state_snapshot)
