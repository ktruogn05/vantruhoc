"""Training and evaluation configuration — dataclass-based, no YAML."""

from dataclasses import dataclass, field

from env.reward import RewardWeights


@dataclass
class WorkloadConfig:
    arrival_rate: float = 0.06
    seed: int = 42
    arrival_horizon: int | None = 5000
    max_arrivals: int | None = None
    prompt_len: tuple[int, int] = (64, 1024)
    response_len: tuple[int, int] = (32, 512)
    priority_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    deadline_slack: tuple[int, int] = (50, 200)


@dataclass
class TrainingConfig:
    # Workload
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # MaskablePPO hyperparameters
    total_timesteps: int = 100_000
    learning_rate: float = 5e-5
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    net_arch: list[int] = field(default_factory=lambda: [256, 256])

    # Logging
    log_dir: str = "logs"
    save_dir: str = "models"
    save_freq: int = 50_000
    log_interval: int = 1
    eval_freq: int = 25_000
    n_eval_episodes: int = 5
    eval_seed: int = 10_000


@dataclass
class EvalConfig:
    n_episodes: int = 10
    seed: int = 123
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
