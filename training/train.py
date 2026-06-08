"""Training pipeline - MaskablePPO on LLMEnvSimple."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-cache"),
)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.default import TrainingConfig
from env.sb3_wrapper import SB3EnvWrapper
from training.env_factory import build_sb3_env


def make_env(config: TrainingConfig, seed: int | None = None) -> SB3EnvWrapper:
    """Create wrapped environment from config."""
    return build_sb3_env(config.workload, config.reward_weights, seed=seed)


def train(config: TrainingConfig | None = None) -> Path:
    """Train MaskablePPO and return the saved model path."""
    if config is None:
        config = TrainingConfig()

    save_dir = Path(config.save_dir)
    log_dir = Path(config.log_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(make_env(config))])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    eval_env = DummyVecEnv([lambda: Monitor(make_env(config, seed=config.eval_seed))])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

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
        tensorboard_log=str(log_dir),
    )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=config.save_freq,
                save_path=str(save_dir),
                name_prefix="llm_scheduler",
            ),
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(save_dir / "best"),
                log_path=str(log_dir / "eval"),
                eval_freq=config.eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                deterministic=True,
            ),
        ]
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=config.log_interval,
    )

    save_path = save_dir / "llm_scheduler_final"
    model.save(str(save_path))
    env.save(str(save_dir / "vecnormalize.pkl"))
    print(f"Model saved to {save_path}")
    print(f"VecNormalize stats saved to {save_dir / 'vecnormalize.pkl'}")

    return save_path
