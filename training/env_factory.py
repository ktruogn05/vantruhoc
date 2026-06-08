"""Environment construction helpers shared by train/evaluate/CLI."""

from configs.default import WorkloadConfig
from core.workload import WorkloadGenerator
from env.llm_env import LLMEnvSimple
from env.reward import RewardWeights
from env.sb3_wrapper import SB3EnvWrapper


def build_workload(config: WorkloadConfig, seed: int | None = None) -> WorkloadGenerator:
    """Create a workload generator with all configured distributions applied."""
    wg = WorkloadGenerator(
        arrival_rate=config.arrival_rate,
        seed=config.seed if seed is None else seed,
        arrival_horizon=config.arrival_horizon,
        max_arrivals=config.max_arrivals,
    )
    wg.set_distributions(
        prompt_len=config.prompt_len,
        response_len=config.response_len,
        priority_weights=config.priority_weights,
        deadline_slack=config.deadline_slack,
    )
    return wg


def build_raw_env(
    workload: WorkloadConfig,
    reward_weights: RewardWeights,
    seed: int | None = None,
) -> LLMEnvSimple:
    return LLMEnvSimple(
        workload_generator=build_workload(workload, seed=seed),
        reward_weights=reward_weights,
    )


def build_sb3_env(
    workload: WorkloadConfig,
    reward_weights: RewardWeights,
    seed: int | None = None,
) -> SB3EnvWrapper:
    return SB3EnvWrapper(build_raw_env(workload, reward_weights, seed=seed))
