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
        from core.workload import WorkloadGenerator
        from env.llm_env import LLMEnvSimple
        from scheduler.rl_agent import RLScheduler
        from training.evaluate import evaluate_scheduler

        config = EvalConfig()
        wg = WorkloadGenerator(
            arrival_rate=config.workload.arrival_rate, seed=config.workload.seed
        )
        env = LLMEnvSimple(workload_generator=wg, reward_weights=config.reward_weights)
        scheduler = RLScheduler(model_path=model_path)
        metrics = evaluate_scheduler(scheduler, env, n_episodes=config.n_episodes)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    elif command == "compare":
        from scheduler.fcfs import FCFSScheduler
        from scheduler.edf import EDFScheduler
        from scheduler.priority import PriorityScheduler
        from training.evaluate import compare_all

        schedulers = {
            "FCFS": FCFSScheduler(),
            "EDF": EDFScheduler(),
            "Priority": PriorityScheduler(),
        }

        # If model path provided, add RL scheduler
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
    """Quick sanity check: random policy for 1 episode."""
    import numpy as np
    from core.workload import WorkloadGenerator
    from env.llm_env import LLMEnvSimple
    from env.sb3_wrapper import SB3EnvWrapper

    print("Creating environment...")
    wg = WorkloadGenerator(arrival_rate=2.0, seed=42)
    raw_env = LLMEnvSimple(workload_generator=wg)
    env = SB3EnvWrapper(raw_env)

    print("Running random policy for 1 episode...")
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0

    while True:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = int(np.random.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 500 == 0:
            print(f"  Step {steps}: reward={reward:.2f}, cumulative={total_reward:.2f}")

        if terminated or truncated:
            break

    print(f"\nEpisode done in {steps} steps.")
    print(f"Total reward: {total_reward:.2f}")
    if info.get("episode_metrics"):
        print("Episode metrics:")
        for k, v in info["episode_metrics"].items():
            print(f"  {k}: {v}")

    # Verify basics
    assert np.isfinite(total_reward), "Reward is not finite!"
    assert steps <= 5000, f"Episode exceeded max steps: {steps}"
    print("\n[OK] Smoke test PASSED")


if __name__ == "__main__":
    main()
