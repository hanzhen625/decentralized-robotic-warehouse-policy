"""Simulate two robots using random actions or a saved decentralized PPO model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from warehouse_rewards import make_env
from action_selection import select_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true", help="Run without a window")
    parser.add_argument("--model", help="Saved PPO checkpoint, e.g. results/decentralized/model.zip")
    parser.add_argument("--seed", type=int, default=10000, help="Warehouse reset seed")
    parser.add_argument("--reward-system", choices=["baseline", "delivery", "shaped", "local_return"],
                        help="Automatically read from the model's config.json when available")
    parser.add_argument("--action-selection", choices=["sampled", "greedy", "mixed"],
                        help="Defaults to checkpoint configuration; sampled for local_return")
    parser.add_argument("--sample-probability", type=float,
                        help="In mixed mode, chance of sampling per robot/step (default: 0.1)")
    args = parser.parse_args()

    model = None
    if args.model:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        model = PPO.load(args.model, device="cpu")
        torch.manual_seed(args.seed)  # PPO.load restores its training seed first.

    saved = Path(args.model).parent / "config.json" if args.model else None
    config = json.loads(saved.read_text()) if saved and saved.exists() else {}
    reward_system = args.reward_system or config.get("reward_system", "local_return" if model is None else "baseline")
    action_selection = args.action_selection or config.get("action_selection", (
        "sampled" if reward_system == "local_return" else "greedy"))
    probability = (args.sample_probability if args.sample_probability is not None
                   else config.get("sample_probability", 0.1))
    if not 0 <= probability <= 1:
        parser.error("sample-probability must be between 0 and 1")
    rng = np.random.default_rng(args.seed)
    env = make_env(reward_system, render_mode=None if args.no_render else "human")
    try:
        observations, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        if not args.no_render:
            env.render()

        for step in range(1000):
            # Batched inference: each robot still uses only its own observation.
            actions = (env.action_space.sample() if model is None else
                       select_actions(model, observations, action_selection, rng, probability))
            observations, rewards, terminated, truncated, info = env.step(actions)
            print(f"Step {step + 1}: Actions={actions}, Rewards={rewards}")

            if not args.no_render:
                env.render()
                time.sleep(0.01)

            if terminated or truncated:
                break
        metrics = env.unwrapped.task_metrics()
        print(f"Finished: {metrics['deliveries']}/2 delivered, "
              f"{metrics['returned']}/2 returned, {metrics['shelves_away']} shelves away "
              f"from home; success={metrics['task_complete']}; {step + 1} steps.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
