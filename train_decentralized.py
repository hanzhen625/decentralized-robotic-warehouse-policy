"""Train shared-policy IPPO; evaluate against random actions on held-out seeds."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv

from rware.warehouse import Action
from warehouse_rewards import GAMMA, REWARD_SETTINGS, make_env
from action_selection import select_actions


class RobotVecEnv(VecEnv):
    """Expose robots as PPO slots, stepping their shared warehouse exactly once.

    SB3 keeps each slot's returns separate. Both actor and critic see only that
    robot's observation; sharing weights does not share observations.
    """

    def __init__(self, reward_system="local_return"):
        self.env = make_env(reward_system)
        self.returns = np.zeros(self.env.unwrapped.n_agents)
        self.length = 0
        super().__init__(len(self.returns), self.env.observation_space[0],
                         self.env.action_space[0])

    def reset(self):
        obs, _ = self.env.reset(seed=self._seeds[0])
        self._reset_seeds()
        self.returns[:] = 0
        self.length = 0
        return np.asarray(obs, dtype=np.float32)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, done, truncated, _ = self.env.step(self.actions)
        self.returns += rewards
        self.length += 1
        finished = done or truncated
        infos = [{} for _ in range(self.num_envs)]
        if finished:
            # This fork reports its step limit as termination. Correct it for PPO
            # so unfinished requests at the time limit bootstrap their values.
            success = self.env.unwrapped.task_metrics()["task_complete"]
            timeout = bool(truncated or not success)
            for i, info in enumerate(infos):
                info.update(terminal_observation=np.asarray(obs[i], dtype=np.float32),
                            episode={"r": float(self.returns[i]), "l": self.length})
                info["TimeLimit.truncated"] = timeout
                info["is_success"] = success
            obs = self.reset()
        return (np.asarray(obs, dtype=np.float32), np.asarray(rewards, dtype=np.float32),
                np.full(self.num_envs, finished), infos)

    def close(self):
        self.env.close()

    # VecEnv's small compatibility interface; attributes belong to one warehouse.
    def get_attr(self, name, indices=None):
        return [getattr(self.env.unwrapped, name) for _ in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        setattr(self.env.unwrapped, name, value)

    def env_method(self, name, *args, indices=None, **kwargs):
        raise NotImplementedError("Call methods directly on the shared warehouse.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def evaluate(model, episodes, reward_system="local_return", action_selection="greedy",
             sample_probability=0.1):
    """Use identical layouts/requests for both policies; count actual deliveries."""
    rows = []
    env = make_env(reward_system)
    for policy in ("random", "ippo"):
        for seed in range(10000, 10000 + episodes):
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            torch.manual_seed(seed)  # Reproduce sampled actions independently per episode.
            rng = np.random.default_rng(seed)
            blocked = 0
            for step in range(1, 1001):
                actions = (env.action_space.sample() if policy == "random"
                           else select_actions(model, obs, action_selection, rng, sample_probability))
                positions = [(a.x, a.y) for a in env.unwrapped.agents]
                obs, _, done, truncated, _ = env.step(actions)
                blocked += sum(int(action == Action.FORWARD.value and pos == (a.x, a.y))
                               for action, pos, a in zip(actions, positions, env.unwrapped.agents))
                if done or truncated:
                    break
            metrics = env.unwrapped.task_metrics()
            rows.append(dict(policy=policy, seed=seed, deliveries=metrics["deliveries"],
                             returned=metrics["returned"], holes=metrics["holes"],
                             shelves_away=metrics["shelves_away"],
                             success=int(metrics["task_complete"]), steps=step, blocked=blocked))
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Agent transitions; two transitions = one warehouse step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/local_return"))
    parser.add_argument("--reward-system", choices=["baseline", "delivery", "shaped", "local_return"],
                        help="Defaults to local_return, or the loaded checkpoint's configuration")
    parser.add_argument("--load", type=Path, help="Evaluate a saved model without training")
    parser.add_argument("--action-selection", choices=["sampled", "greedy", "mixed"],
                        help="Evaluation actions; local_return defaults to sampled as in training")
    parser.add_argument("--sample-probability", type=float,
                        help="Mixed-mode sampling probability per robot/step (default: 0.1)")
    args = parser.parse_args()
    if args.steps <= 0 or args.episodes <= 0:
        parser.error("steps and episodes must be positive")
    saved = args.load.parent / "config.json" if args.load else None
    previous = json.loads(saved.read_text()) if saved and saved.exists() else {}
    args.reward_system = args.reward_system or (
        previous.get("reward_system", "baseline") if args.load else "local_return")
    args.action_selection = args.action_selection or previous.get("action_selection", (
        "sampled" if args.reward_system == "local_return" else "greedy"))
    args.sample_probability = (args.sample_probability if args.sample_probability is not None
                               else previous.get("sample_probability", 0.1))
    if not 0 <= args.sample_probability <= 1:
        parser.error("sample-probability must be between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # Small networks run faster without thread overhead.
    started = time.perf_counter()
    if args.load:
        model = PPO.load(args.load, device="cpu")
    else:
        env = RobotVecEnv(args.reward_system)
        model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", n_steps=512,
                    batch_size=256, n_epochs=5, ent_coef=0.02, gamma=GAMMA,
                    policy_kwargs=dict(net_arch=[64, 64]), verbose=0)
        model.set_logger(configure(str(args.output), ["csv"]))
        print(f"Training {args.steps:,} agent transitions...", flush=True)
        model.learn(total_timesteps=args.steps, log_interval=10)
        model.save(args.output / "model")
        env.close()
    seconds = time.perf_counter() - started
    rows = evaluate(model, args.episodes, args.reward_system, args.action_selection,
                    args.sample_probability)
    with (args.output / "evaluation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Decentralized policy results", "",
              "Shared-policy IPPO: separate robot decisions, shared weights and training samples.",
              "Tiny warehouse (1 shelf row, 3 columns, height 8), 2 robots, 2 total deliveries,",
              "Sensor range 1, no messages, normalized coordinates, 1,000-step limit.",
              f"Reward system: {args.reward_system}.",
              ("Success requires both deliveries, every shelf at its original home, and no carried shelves."
               if args.reward_system == "local_return" else "Task ends at the final delivery."),
              ("Baseline: local sensors, TWO_STAGE reward, no target features." if
               args.reward_system == "baseline" else
               ("Local sensors and private memory only; static map/stations and own shelf home metadata. "
                "+1 delivery, +0.5 first home return after delivery, -0.001 per step; "
                "+0.002 per newly seen storage cell (once per robot)."
                if args.reward_system == "local_return" else
                "Local sensors + broadcast destinations/map and own task phase; +1 per delivery, -0.001 per step.")),
              ("Progress shaping: gamma * potential(next) - potential(now), gamma=0.99; "
               "potential=0.5*exp(-remaining_distance/20)." if args.reward_system in ("shaped", "local_return")
               else "No additional progress shaping."), "",
              f"Checkpoint: {model.num_timesteps:,} agent transitions "
              f"({model.num_timesteps // 2:,} warehouse steps); training seed {model.seed}.",
              f"{'Load' if args.load else 'Training'} time: {seconds:.1f} seconds. "
              f"Evaluation: {args.episodes} paired seeds starting at 10000; "
              f"{args.action_selection} IPPO actions (sampling seeded per episode).", "",
              f"Mixed-mode sampling probability: {args.sample_probability:g} per robot/step (used only in mixed mode).", "",
              "| Policy | Full success | Deliveries/episode | Returned/episode | Shelves away/episode | Steps/success |",
              "|---|---:|---:|---:|---:|---:|"]
    for policy in ("random", "ippo"):
        data = [r for r in rows if r["policy"] == policy]
        successes = [r["steps"] for r in data if r["success"]]
        steps = f"{np.mean(successes):.1f}" if successes else "N/A"
        report.append(f"| {policy} | {sum(r['success'] for r in data)}/{len(data)} | "
                      f"{np.mean([r['deliveries'] for r in data]):.2f} | "
                      f"{np.mean([r['returned'] for r in data]):.2f} | "
                      f"{np.mean([r['shelves_away'] for r in data]):.2f} | {steps} |")
    report += ["", "Blocked forwards include wall/shelf obstruction and robot conflicts.",
               "One training seed: preliminary results, not evidence of superiority over centralized control."]
    learned = [r for r in rows if r["policy"] == "ippo"]
    if not any(r["success"] for r in learned):
        report += ["", "Outcome: IPPO did not complete the task on any evaluation seed;",
                   "this checkpoint is an unsuccessful initial training trial, not a reliable controller."]
    (args.output / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    config = {**vars(args), "output": str(args.output),
              "load": str(args.load) if args.load else None,
              "actual_agent_transitions": model.num_timesteps,
              "reward_settings": REWARD_SETTINGS if args.reward_system != "baseline" else None,
              "torch": torch.__version__}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
