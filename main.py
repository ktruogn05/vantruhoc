"""Entry point: train / evaluate / compare."""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|evaluate|compare|smoke-test]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        from training.train import train

        train()

    elif command == "evaluate":
        if len(sys.argv) < 3:
            print("Usage: python main.py evaluate <model_path>")
            sys.exit(1)
        model_path = sys.argv[2]

        from configs.default import EvalConfig
        from scheduler.rl_agent import RLScheduler
        from training.env_factory import build_raw_env
        from training.evaluate import evaluate_scheduler

        config = EvalConfig()
        env = build_raw_env(config.workload, config.reward_weights)
        scheduler = RLScheduler(model_path=model_path)
        metrics = evaluate_scheduler(
            scheduler,
            env,
            n_episodes=config.n_episodes,
            seed_start=config.seed,
        )
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    elif command == "compare":
        from scheduler.edf import EDFScheduler
        from scheduler.fcfs import FCFSScheduler
        from scheduler.priority import PriorityScheduler
        from training.evaluate import compare_all

        schedulers = {
            "FCFS": FCFSScheduler(),
            "EDF": EDFScheduler(),
            "Priority": PriorityScheduler(),
        }

        if len(sys.argv) >= 3:
            from scheduler.rl_agent import RLScheduler

            schedulers["RL"] = RLScheduler(model_path=sys.argv[2])

        df = compare_all(schedulers)
        print("\n=== Comparison Results ===")
        print(df.to_string())

    elif command == "smoke-test":
        _smoke_test()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def _smoke_test():
    """Deterministic quality smoke test on a finite workload."""
    import numpy as np

    from configs.default import WorkloadConfig
    from env.reward import RewardWeights
    from scheduler.fcfs import FCFSScheduler
    from training.env_factory import build_raw_env
    from training.evaluate import evaluate_scheduler

    workload = WorkloadConfig(
        arrival_rate=0.2,
        seed=42,
        arrival_horizon=200,
        max_arrivals=20,
        prompt_len=(64, 256),
        response_len=(16, 64),
        deadline_slack=(200, 400),
    )
    env = build_raw_env(workload, RewardWeights())
    metrics = evaluate_scheduler(FCFSScheduler(), env, n_episodes=1, seed_start=42)

    print("Smoke-test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    assert metrics["total_steps_mean"] <= 5000
    assert np.isfinite(metrics["throughput_tokens_per_step_mean"])
    assert metrics["requests_arrived_mean"] > 0
    assert metrics["completion_rate_mean"] >= 0.95
    assert metrics["requests_aborted_mean"] == 0
    assert metrics["requests_unfinished_mean"] == 0
    print("\n[OK] Smoke test PASSED")


if __name__ == "__main__":
    main()
