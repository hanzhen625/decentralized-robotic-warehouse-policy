
import ast
import csv
import random
from pathlib import Path

import numpy as np

from train_decentralized import make_env, get_state

RESULTS = Path("decentralized_results")
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005]
MAX_STEPS = 200


def load_q_table(robot_number):
    path = RESULTS / f"robot_{robot_number}_q_table.csv"
    table = {}

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            state = ast.literal_eval(row["state"])
            values = [
                float(row[f"action_{i}"])
                for i in range(5)
            ]
            table[state] = values

    return table


def choose_action(table, state):
    # No exploration during evaluation.
    # For unseen states, use NOOP as a fallback.
    if state not in table:
        return 0, False

    values = table[state]
    best = max(values)

    # Deterministic tie-breaking.
    action = values.index(best)
    return action, True


def evaluate_episode(tables, seed):
    env = make_env()
    env.reset(seed=seed)

    total_reward = 0.0
    robot_rewards = [0.0, 0.0]
    visited = 0
    known = 0
    movements = 0
    steps = 0

    try:
        for step in range(MAX_STEPS):
            states = [
                get_state(agent, env)
                for agent in env.agents
            ]

            old_positions = [
                (int(agent.x), int(agent.y))
                for agent in env.agents
            ]

            decisions = [
                choose_action(tables[i], states[i])
                for i in range(2)
            ]

            actions = [item[0] for item in decisions]
            known += sum(item[1] for item in decisions)
            visited += 2

            _, rewards, terminated, truncated, _ = env.step(actions)

            for i, reward in enumerate(rewards):
                robot_rewards[i] += float(reward)

            new_positions = [
                (int(agent.x), int(agent.y))
                for agent in env.agents
            ]

            movements += sum(
                old != new
                for old, new in zip(old_positions, new_positions)
            )

            steps = step + 1

            done = (
                all(terminated)
                if hasattr(terminated, "__iter__")
                else bool(terminated)
            )

            timeout = (
                all(truncated)
                if hasattr(truncated, "__iter__")
                else bool(truncated)
            )

            if done or timeout:
                break

    finally:
        env.close()

    total_reward = sum(robot_rewards)

    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "robot_1_reward": robot_rewards[0],
        "robot_2_reward": robot_rewards[1],
        "movement_actions": movements,
        "known_state_fraction": known / max(visited, 1),
    }


def main():
    tables = [
        load_q_table(1),
        load_q_table(2),
    ]

    print("Evaluating trained decentralized Q-learning...")
    print("Exploration: OFF")
    print("Robot 1 Q-table states:", len(tables[0]))
    print("Robot 2 Q-table states:", len(tables[1]))

    rows = []

    for seed in TEST_SEEDS:
        result = evaluate_episode(tables, seed)
        rows.append(result)

        print(
            f"Seed {seed}: "
            f"Reward={result['total_reward']:.2f}, "
            f"Moves={result['movement_actions']}, "
            f"Known states={result['known_state_fraction']:.1%}",
            flush=True,
        )

    with (RESULTS / "evaluation_results.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_reward = np.mean([
        row["total_reward"] for row in rows
    ])

    print("\nEvaluation complete!")
    print(f"Mean cumulative reward: {mean_reward:.3f}")
    print(
        "Saved:",
        RESULTS / "evaluation_results.csv"
    )


if __name__ == "__main__":
    main()
