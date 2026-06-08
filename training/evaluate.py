"""Evaluation harness - run schedulers on shared workload configuration."""

from __future__ import annotations

from numbers import Number

import pandas as pd

from configs.default import EvalConfig
from core.types import SchedulerAction
from env.llm_env import LLMEnvSimple
from scheduler.base import BaseScheduler
from training.env_factory import build_raw_env


def evaluate_scheduler(
    scheduler: BaseScheduler,
    env: LLMEnvSimple,
    n_episodes: int = 10,
    seed_start: int = 0,
) -> dict:
    """Run scheduler on env for n_episodes and return aggregated metrics."""
    all_summaries = []

    for ep in range(n_episodes):
        env.reset(seed=seed_start + ep)
        scheduler.on_episode_start()

        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            state = env.state_snapshot
            action: SchedulerAction = scheduler.select_action(state)
            _, _, terminated, truncated, info = env.step(action)

        episode_metrics = info.get("episode_metrics") or {}
        scheduler.on_episode_end(episode_metrics)
        all_summaries.append(episode_metrics)

    return aggregate_metrics(all_summaries)


def aggregate_metrics(summaries: list[dict]) -> dict:
    if not summaries:
        return {}

    keys = sorted({key for summary in summaries for key in summary.keys()})
    aggregated = {}
    for key in keys:
        values = [
            summary[key]
            for summary in summaries
            if isinstance(summary.get(key), Number)
        ]
        if not values:
            continue
        aggregated[f"{key}_mean"] = sum(values) / len(values)
        aggregated[f"{key}_min"] = min(values)
        aggregated[f"{key}_max"] = max(values)
    return aggregated


def compare_all(
    schedulers: dict[str, BaseScheduler],
    config: EvalConfig | None = None,
) -> pd.DataFrame:
    """Run all schedulers on the same workload config."""
    if config is None:
        config = EvalConfig()

    results = {}
    for name, scheduler in schedulers.items():
        env = build_raw_env(config.workload, config.reward_weights)
        metrics = evaluate_scheduler(
            scheduler,
            env,
            n_episodes=config.n_episodes,
            seed_start=config.seed,
        )
        results[name] = metrics
        completed = metrics.get("requests_completed_mean", "N/A")
        print(f"[{name}] done - {completed} avg completed")

    return pd.DataFrame(results).T
