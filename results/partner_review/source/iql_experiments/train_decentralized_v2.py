
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rware.warehouse import Warehouse, RewardType, ObservationType


# ==========================================
# SETTINGS
# ==========================================

EPISODES = 2000
MAX_STEPS = 200

LEARNING_RATE = 0.15
DISCOUNT = 0.95

EPSILON_START = 1.0
EPSILON_END = 0.05

OUTPUT = Path("decentralized_v2_results")

NOOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
TOGGLE = 4

ACTIONS = [NOOP, FORWARD, LEFT, RIGHT, TOGGLE]


# ==========================================
# WAREHOUSE ENVIRONMENT
# ==========================================

def make_env():
    return Warehouse(
        shelf_rows=1,
        shelf_columns=3,
        column_height=8,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=2,
        max_inactivity_steps=None,
        max_steps=MAX_STEPS,
        reward_type=RewardType.INDIVIDUAL,
        observation_type=ObservationType.FLATTENED,
        render_mode=None,
    )


# ==========================================
# STATE REPRESENTATION
# ==========================================

def get_state(agent, env):
    x = int(agent.x)
    y = int(agent.y)
    direction = int(agent.dir.value)

    carrying = int(agent.carrying_shelf is not None)

    if carrying:
        targets = [
            tuple(map(int, goal))
            for goal in env.goals
        ]
    else:
        targets = [
            (int(shelf.x), int(shelf.y))
            for shelf in env.request_queue
        ]

    if targets:
        tx, ty = min(
            targets,
            key=lambda t: abs(t[0] - x) + abs(t[1] - y)
        )
    else:
        tx, ty = x, y

    dx = (tx > x) - (tx < x)
    dy = (ty > y) - (ty < y)

    return (x, y, direction, carrying, dx, dy)


def distance_to_target(agent, env):
    x, y = int(agent.x), int(agent.y)

    if agent.carrying_shelf is not None:
        targets = [
            tuple(map(int, goal))
            for goal in env.goals
        ]
    else:
        targets = [
            (int(shelf.x), int(shelf.y))
            for shelf in env.request_queue
        ]

    if not targets:
        return 0

    return min(
        abs(tx - x) + abs(ty - y)
        for tx, ty in targets
    )


# ==========================================
# Q-LEARNING
# ==========================================

def make_q_table():
    return defaultdict(lambda: [0.0] * len(ACTIONS))


def choose_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)

    values = q_table[state]
    best_value = max(values)

    best_actions = [
        action
        for action, value in enumerate(values)
        if value == best_value
    ]

    # Random tie-breaking prevents automatic NOOP.
    return rng.choice(best_actions)


def train():
    OUTPUT.mkdir(exist_ok=True)

    rng = random.Random(42)

    q_tables = [
        make_q_table(),
        make_q_table(),
    ]

    history = []

    print("Starting Independent Q-Learning V2...", flush=True)

    for episode in range(EPISODES):
        env = make_env()
        env.reset(seed=episode + 42)

        epsilon = max(
            EPSILON_END,
            EPSILON_START
            - (EPSILON_START - EPSILON_END)
            * episode / EPISODES
        )

        total_environment_reward = 0.0
        total_shaped_reward = 0.0

        try:
            for step in range(MAX_STEPS):

                states = [
                    get_state(agent, env)
                    for agent in env.agents
                ]

                old_distances = [
                    distance_to_target(agent, env)
                    for agent in env.agents
                ]

                was_carrying = [
                    agent.carrying_shelf is not None
                    for agent in env.agents
                ]

                actions = [
                    choose_action(
                        q_tables[i],
                        states[i],
                        epsilon,
                        rng
                    )
                    for i in range(2)
                ]

                _, rewards, terminated, truncated, _ = env.step(actions)

                next_states = [
                    get_state(agent, env)
                    for agent in env.agents
                ]

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

                for i in range(2):
                    agent = env.agents[i]

                    environment_reward = float(rewards[i])

                    shaped_reward = environment_reward

                    # Small step penalty to discourage wasting time.
                    shaped_reward -= 0.01

                    # Encourage movement toward a target.
                    new_distance = distance_to_target(agent, env)

                    if new_distance < old_distances[i]:
                        shaped_reward += 0.05
                    elif new_distance > old_distances[i]:
                        shaped_reward -= 0.02

                    # Encourage picking up a shelf.
                    now_carrying = agent.carrying_shelf is not None

                    if not was_carrying[i] and now_carrying:
                        shaped_reward += 0.5

                    # Update Q-values.
                    old_q = q_tables[i][states[i]][actions[i]]

                    future_q = (
                        0.0 if done or timeout
                        else max(q_tables[i][next_states[i]])
                    )

                    target = (
                        shaped_reward
                        + DISCOUNT * future_q
                    )

                    q_tables[i][states[i]][actions[i]] = (
                        old_q
                        + LEARNING_RATE * (target - old_q)
                    )

                    total_environment_reward += environment_reward
                    total_shaped_reward += shaped_reward

                if done or timeout:
                    break

        finally:
            env.close()

        history.append({
            "episode": episode + 1,
            "environment_reward": total_environment_reward,
            "shaped_reward": total_shaped_reward,
            "epsilon": epsilon,
        })

        if (episode + 1) % 100 == 0:
            recent = history[-100:]

            avg_env = sum(
                row["environment_reward"]
                for row in recent
            ) / len(recent)

            avg_shaped = sum(
                row["shaped_reward"]
                for row in recent
            ) / len(recent)

            print(
                f"Episode {episode + 1}/{EPISODES} "
                f"| Delivery reward: {avg_env:.3f} "
                f"| Shaped reward: {avg_shaped:.3f} "
                f"| Epsilon: {epsilon:.3f}",
                flush=True,
            )

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    with (OUTPUT / "training_rewards.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "environment_reward",
                "shaped_reward",
                "epsilon",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    for i, table in enumerate(q_tables):
        with (
            OUTPUT / f"robot_{i + 1}_q_table.csv"
        ).open("w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(
                ["state"]
                + [f"action_{a}" for a in ACTIONS]
            )

            for state, values in table.items():
                writer.writerow([str(state)] + values)

    # Plot environment reward only.
    moving_average = []

    for i in range(len(history)):
        recent = history[max(0, i - 99):i + 1]

        moving_average.append(
            sum(
                row["environment_reward"]
                for row in recent
            ) / len(recent)
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        range(1, EPISODES + 1),
        moving_average,
    )

    ax.set_title(
        "Independent Q-Learning V2: Delivery Reward"
    )
    ax.set_xlabel("Training Episode")
    ax.set_ylabel(
        "Mean Environment Reward (Last 100 Episodes)"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "training_curve.png",
        dpi=180
    )

    plt.close(fig)

    print("\nV2 training complete!")
    print("Saved to:", OUTPUT.resolve())
    print("Robot 1 states:", len(q_tables[0]))
    print("Robot 2 states:", len(q_tables[1]))


if __name__ == "__main__":
    train()
