"""Evaluation harness — run any scheduler on env, compare results."""

from __future__ import annotations

import pandas as pd

from configs.default import EvalConfig
from core.types import SchedulerAction, ActionType
from core.workload import WorkloadGenerator
from env.llm_env import LLMEnvSimple
from scheduler.base import BaseScheduler


def evaluate_scheduler(
    scheduler: BaseScheduler,
    env: LLMEnvSimple,
    n_episodes: int = 10,
    seed_start: int = 0,
) -> dict:
    """Run scheduler on env for n_episodes. Return aggregated metrics."""
    all_summaries = []

    for ep in range(n_episodes):
        env.reset(seed=seed_start + ep)
        scheduler.on_episode_start()

        terminated = False
        truncated = False

        while not (terminated or truncated):
            state = env.state_snapshot
            action: SchedulerAction = scheduler.select_action(state)
            _, _, terminated, truncated, info = env.step(action)

        episode_metrics = info.get("episode_metrics", {})
        scheduler.on_episode_end(episode_metrics)
        all_summaries.append(episode_metrics)

    # Aggregate across episodes
    if not all_summaries:
        return {}

    keys = all_summaries[0].keys()
    aggregated = {}
    for key in keys:
        values = [s[key] for s in all_summaries if s is not None]
        if values:
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)

    return aggregated


def compare_all(
    schedulers: dict[str, BaseScheduler],
    config: EvalConfig | None = None,
) -> pd.DataFrame:
    """Run all schedulers on same workload config. Return comparison DataFrame."""
    if config is None:
        config = EvalConfig()

    results = {}

    for name, scheduler in schedulers.items():
        # Create fresh env for each scheduler (same config)
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
        env = LLMEnvSimple(
            workload_generator=wg,
            reward_weights=config.reward_weights,
        )

        metrics = evaluate_scheduler(
            scheduler, env, n_episodes=config.n_episodes, seed_start=config.seed
        )
        results[name] = metrics
        print(f"[{name}] done — {metrics.get('requests_completed_mean', 'N/A')} avg completed")

    df = pd.DataFrame(results).T
    return df
