"""Training pipeline — MaskablePPO on LLMEnvSimple."""

from __future__ import annotations

import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from configs.default import TrainingConfig
from core.workload import WorkloadGenerator
from env.llm_env import LLMEnvSimple
from env.sb3_wrapper import SB3EnvWrapper


def make_env(config: TrainingConfig) -> SB3EnvWrapper:
    """Create wrapped environment from config."""
    wg = WorkloadGenerator(
        arrival_rate=config.workload.arrival_rate,
        seed=config.workload.seed,
    )
    wg.set_distributions(
        prompt_len=config.workload.prompt_len,
        response_len=config.workload.response_len,
        priority_weights=config.workload.priority_weights,
        deadline_slack=config.workload.deadline_slack,
    )
    raw_env = LLMEnvSimple(
        workload_generator=wg,
        reward_weights=config.reward_weights,
    )
    return SB3EnvWrapper(raw_env)


def train(config: TrainingConfig | None = None) -> Path:
    """Train MaskablePPO. Returns path to saved model."""
    if config is None:
        config = TrainingConfig()

    # Wrap in standard DummyVecEnv and VecNormalize for stable dynamic reward scaling
    env = DummyVecEnv([lambda: make_env(config)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": config.net_arch},
        verbose=1,
        tensorboard_log=config.log_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.save_dir,
        name_prefix="llm_scheduler",
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    save_path = Path(config.save_dir) / "llm_scheduler_final"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    return save_path
