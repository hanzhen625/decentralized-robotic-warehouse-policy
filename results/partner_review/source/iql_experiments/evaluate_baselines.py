
import csv
import random
from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rware.warehouse import Warehouse, RewardType, ObservationType


# Experiment settings
SEEDS = [42, 43, 44, 45, 46]
MAX_STEPS = 500
OUTPUT = Path("baseline_results")

# RWARE actions
NOOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
TOGGLE = 4

# RWARE directions
DELTAS = {
    0: (0, -1),  # Up
    1: (0, 1),   # Down
    2: (-1, 0),  # Left
    3: (1, 0),   # Right
}
CLOCKWISE = [0, 3, 1, 2]


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


def pos(obj):
    return (int(obj.x), int(obj.y))


def path_to(start, target, blocked, grid_size):
    height, width = map(int, grid_size)

    queue = deque([start])
    previous = {start: None}

    while queue:
        current = queue.popleft()

        if current == target:
            break

        for dx, dy in DELTAS.values():
            nxt = (
                current[0] + dx,
                current[1] + dy,
            )

            if (
                0 <= nxt[0] < width
                and 0 <= nxt[1] < height
                and nxt not in blocked
                and nxt not in previous
            ):
                previous[nxt] = current
                queue.append(nxt)

    if target not in previous:
        return None

    path = []
    current = target

    while current is not None:
        path.append(current)
        current = previous[current]

    return path[::-1]


def blocked_cells(env, agent):
    blocked = {
        pos(other)
        for other in env.agents
        if other is not agent
    }

    # Carrying robots cannot move through other shelves.
    if agent.carrying_shelf is not None:
        blocked.update(
            pos(shelf)
            for shelf in env.shelfs
            if shelf is not agent.carrying_shelf
        )

    return blocked


def steer(agent, next_cell):
    here = pos(agent)

    delta = (
        next_cell[0] - here[0],
        next_cell[1] - here[1],
    )

    desired = next(
        direction
        for direction, vector in DELTAS.items()
        if vector == delta
    )

    current = agent.dir.value

    turn = (
        CLOCKWISE.index(desired)
        - CLOCKWISE.index(current)
    ) % 4

    return {
        0: FORWARD,
        1: RIGHT,
        2: RIGHT,
        3: LEFT,
    }[turn]


def choose_heuristic_action(env, agent, state):
    # Return delivered shelf to its original location.
    if state["phase"] == "return" and state["home"] is not None:
        candidates = [state["home"]]

    # Carry shelf to a delivery goal.
    elif agent.carrying_shelf is not None:
        state["phase"] = "deliver"
        candidates = [
            tuple(map(int, goal))
            for goal in env.goals
        ]

    # Find a requested shelf.
    else:
        state["phase"] = "pickup"
        candidates = [
            pos(shelf)
            for shelf in env.request_queue
        ]

    blocked = blocked_cells(env, agent)

    paths = [
        path_to(pos(agent), target, blocked, env.grid_size)
        for target in candidates
    ]

    valid = [
        (len(path), target, path)
        for target, path in zip(candidates, paths)
        if path is not None
    ]

    if not valid:
        return NOOP

    _, target, path = min(valid, key=lambda item: item[0])

    if pos(agent) == target:
        if state["phase"] == "pickup":
            state["home"] = target
            return TOGGLE

        if state["phase"] == "return":
            state["phase"] = "pickup"
            state["home"] = None
            return TOGGLE

        return NOOP

    return steer(agent, path[1])


def run_episode(policy, seed):
    env = make_env()
    rng = random.Random(seed)

    env.reset(seed=seed)

    agent = env.agents[0]

    state = {
        "phase": "pickup",
        "home": None,
    }

    rewards_total = [0.0, 0.0]
    rewarded_steps = 0
    steps = 0

    try:
        for _ in range(MAX_STEPS):

            if policy == "random":
                actions = [
                    rng.randrange(5),
                    rng.randrange(5),
                ]

            elif policy == "heuristic":
                actions = [
                    choose_heuristic_action(env, agent, state),
                    NOOP,
                ]

            else:
                raise ValueError(f"Unknown policy: {policy}")

            _, rewards, terminated, truncated, _ = env.step(actions)

            steps += 1

            reward_values = [float(r) for r in rewards]

            rewards_total = [
                a + b
                for a, b in zip(rewards_total, reward_values)
            ]

            if any(r > 0 for r in reward_values):
                rewarded_steps += 1

                if policy == "heuristic":
                    state["phase"] = "return"

            done = (
                all(terminated)
                if hasattr(terminated, "__iter__")
                else terminated
            )

            timeout = (
                all(truncated)
                if hasattr(truncated, "__iter__")
                else truncated
            )

            if done or timeout:
                break

    finally:
        env.close()

    total_reward = sum(rewards_total)

    return {
        "policy": policy,
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "robot_1_reward": rewards_total[0],
        "robot_2_reward": rewards_total[1],
        "rewarded_steps": rewarded_steps,
        "reward_per_100_steps": (
            100 * total_reward / max(steps, 1)
        ),
    }


def main():
    OUTPUT.mkdir(exist_ok=True)

    rows = []

    print("Starting RWARE baseline evaluation...", flush=True)

    # Evaluate both baseline policies.
    for policy in ("random", "heuristic"):
        for seed in SEEDS:

            result = run_episode(policy, seed)
            rows.append(result)

            print(
                f"{policy:9s} "
                f"seed={seed} "
                f"steps={result['steps']} "
                f"reward={result['total_reward']:.2f} "
                f"rewarded_steps={result['rewarded_steps']}",
                flush=True,
            )

    # Save episode-level results.
    with (OUTPUT / "episode_results.csv").open(
        "w", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    # Calculate averages across seeds.
    for policy in ("random", "heuristic"):

        subset = [
            row for row in rows
            if row["policy"] == policy
        ]

        summary.append({
            "policy": policy,
            "episodes": len(subset),
            "mean_total_reward": (
                sum(r["total_reward"] for r in subset)
                / len(subset)
            ),
            "mean_reward_per_100_steps": (
                sum(r["reward_per_100_steps"] for r in subset)
                / len(subset)
            ),
            "mean_rewarded_steps": (
                sum(r["rewarded_steps"] for r in subset)
                / len(subset)
            ),
        })

    # Save summary.
    with (OUTPUT / "summary.csv").open(
        "w", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)

    # Plot mean cumulative reward.
    fig, ax = plt.subplots(figsize=(8, 5))

    policies = [r["policy"] for r in summary]
    means = [r["mean_total_reward"] for r in summary]

    ax.bar(policies, means)
    ax.set_title("RWARE Baseline Policy Comparison")
    ax.set_xlabel("Baseline Policy")
    ax.set_ylabel("Mean Cumulative Reward")

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "baseline_rewards.png",
        dpi=180,
    )

    plt.close(fig)

    print("\nEvaluation complete!")
    print("Files saved to:", OUTPUT.resolve())

    for row in summary:
        print(
            f"{row['policy']}: "
            f"mean reward = {row['mean_total_reward']:.2f}"
        )


if __name__ == "__main__":
    main()
