"""Reproduce partner defaults and compare fixed delivery/return task outcomes.

Run from the repository root after the two unmodified training scripts finish.
The Q tables and BFS receive oracle state as in the submitted code; PPO retains
local sensing/private memory. This is a submitted-system comparison, not an
equal-information/equal-training-budget algorithm benchmark.
"""

import ast
import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from warehouse_rewards import make_env
from action_selection import select_actions


def load_module(name, filename):
    path = HERE / "source/iql_experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tables(folder):
    result = []
    for robot in (1, 2):
        with (HERE / folder / f"robot_{robot}_q_table.csv").open() as f:
            result.append({ast.literal_eval(r["state"]):
                           [float(r[f"action_{a}"]) for a in range(5)]
                           for r in csv.DictReader(f)})
    return result


def q_action(table, state):
    # Exactly the submitted evaluator: unseen states NOOP; first maximum wins.
    values = table.get(state)
    return 0 if values is None else values.index(max(values))


def pickup_loop_audit(v2):
    env = v2.make_env()
    env.reset(seed=42)
    agent, other = env.agents
    shelf = env.request_queue[0]
    agent.x, agent.y = shelf.x, shelf.y
    other.x, other.y = 0, 0
    env._recalc_grid()
    shaped_total, deliveries = 0.0, 0.0
    for _ in range(10):
        old_distance = v2.distance_to_target(agent, env)
        carrying = agent.carrying_shelf is not None
        _, rewards, _, _, _ = env.step([4, 0])
        new_distance = v2.distance_to_target(agent, env)
        reward = float(rewards[0]) - 0.01
        reward += 0.05 if new_distance < old_distance else (-0.02 if new_distance > old_distance else 0)
        if not carrying and agent.carrying_shelf is not None:
            reward += 0.5
        shaped_total += reward
        deliveries += float(rewards[0])
    env.close()
    return dict(actions=[4] * 10, shaped_reward=shaped_total,
                actual_delivery_reward=deliveries,
                carrying_at_end=agent.carrying_shelf is not None)


def main(extra_q_exploration=False):
    torch.set_num_threads(1)
    v1 = load_module("partner_v1", "train_decentralized.py")
    v2 = load_module("partner_v2", "train_decentralized_v2.py")
    bfs = load_module("partner_baseline", "evaluate_baselines.py")
    versions = [("iql_v1", v1, load_tables("decentralized_results")),
                ("iql_v2", v2, load_tables("decentralized_v2_results"))]
    audit = pickup_loop_audit(v2)
    (HERE / "reward_loop_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Pickup/drop loop audit:", audit, flush=True)

    # Reproduce the submitted 200-step evaluation for both tables. The source
    # evaluator only loads V1; use the same decision rule and factory for V2.
    original = []
    for label, module, tables in versions:
        for seed in range(1001, 1006):
            env = module.make_env()
            env.reset(seed=seed)
            total, moved, known = 0.0, 0, 0
            for _ in range(200):
                states = [module.get_state(a, env) for a in env.agents]
                actions = [q_action(t, s) for t, s in zip(tables, states)]
                known += sum(s in t for s, t in zip(states, tables))
                before = [(a.x, a.y) for a in env.agents]
                _, rewards, done, truncated, _ = env.step(actions)
                total += sum(rewards)
                moved += sum(p != (a.x, a.y) for p, a in zip(before, env.agents))
                if done or truncated:
                    break
            env.close()
            original.append(dict(policy=label, seed=seed, delivery_reward=total,
                                 movements=moved, known_state_fraction=known / 400))
    with (HERE / "original_task_evaluation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(original[0]))
        writer.writeheader()
        writer.writerows(original)
    print("Original 200-step evaluations:", original, flush=True)

    model = PPO.load(ROOT / "results/local_return/model.zip", device="cpu")
    policies = [(name, module, tables) for name, module, tables in versions]
    policies += [("bfs_oracle_one_robot", bfs, None), ("ppo_sampled", None, None),
                 ("ppo_mixed_90_10", None, None)]
    if extra_q_exploration:
        policies = [(name + "_epsilon005", module, tables)
                    for name, module, tables in versions]
        with (HERE / "fixed_task_evaluation.csv").open() as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    for label, module, tables in policies:
        for seed in range(10000, 10050):
            env = make_env("local_return")
            obs, _ = env.reset(seed=seed)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            q_rng = random.Random(seed)
            world = env.unwrapped
            state = dict(phase="pickup", home=None)
            for step in range(1, 1001):
                if tables is not None:
                    states = [module.get_state(a, world) for a in world.agents]
                    if extra_q_exploration:
                        actions = []
                        for table, q_state in zip(tables, states):
                            # Source training's final epsilon and random tie rule;
                            # evaluation does not update or insert Q values.
                            if q_rng.random() < 0.05:
                                actions.append(q_rng.choice(range(5)))
                            else:
                                values = table.get(q_state, [0.0] * 5)
                                actions.append(q_rng.choice([a for a, v in enumerate(values)
                                                             if v == max(values)]))
                    else:
                        actions = [q_action(t, s) for t, s in zip(tables, states)]
                elif label.startswith("bfs"):
                    actions = [module.choose_heuristic_action(world, world.agents[0], state), 0]
                else:
                    actions = select_actions(model, obs,
                                             "mixed" if label == "ppo_mixed_90_10" else "sampled",
                                             rng, 0.1)
                obs, _, done, truncated, info = env.step(actions)
                if label.startswith("bfs") and info["base_rewards"][0] > 0:
                    state["phase"] = "return"
                if done or truncated:
                    break
            metrics = world.task_metrics()
            rows.append(dict(policy=label, seed=seed, steps=step, **metrics))
            env.close()
        subset = [r for r in rows if r["policy"] == label]
        print(f"{label}: {sum(r['task_complete'] for r in subset)}/50 full success; "
              f"{np.mean([r['deliveries'] for r in subset]):.2f} deliveries/episode; "
              f"{np.mean([r['returned'] for r in subset]):.2f} returned/episode", flush=True)
        with (HERE / "fixed_task_evaluation.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-q-exploration", action="store_true",
                        help="Append Q-table evaluations with epsilon=.05 and random ties")
    main(parser.parse_args().extra_q_exploration)
