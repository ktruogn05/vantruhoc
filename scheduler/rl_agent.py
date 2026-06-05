"""RL scheduler — wraps SB3 MaskablePPO model behind BaseScheduler interface."""

from __future__ import annotations

from pathlib import Path

from core.types import SchedulerAction
from codec.obs_encoder import ObservationEncoder
from codec.action_decoder import ActionDecoder
from scheduler.base import BaseScheduler


class RLScheduler(BaseScheduler):
    """Adapter: SB3 MaskablePPO → BaseScheduler.

    Model loaded externally. Uses encoder/decoder internally.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.encoder = ObservationEncoder()
        self.decoder = ActionDecoder()
        self.model = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, path: str | Path) -> None:
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(str(path))

    def select_action(self, state_snapshot: dict) -> SchedulerAction:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        obs = self.encoder.encode(state_snapshot)
        mask = self.decoder.get_action_mask(state_snapshot)
        action_id, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True
        )
        return self.decoder.decode(int(action_id))
