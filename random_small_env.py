"""Run a short random-policy demo of the two-robot warehouse."""

import argparse
import time

import gymnasium as gym
import rware  # noqa: F401 - registers the warehouse environments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true", help="Run without a window")
    args = parser.parse_args()

    env = gym.make(
    "rware-tiny-2ag-v2",
    shelf_rows=1,       # number of shelf-block rows
    shelf_columns=3,    # number of shelf-block columns (must be odd)
    column_height=8,    # height of each shelf block in cells
    render_mode="human",
    max_steps=1000,
)
    try:
        env.reset(seed=42)
        env.action_space.seed(42)
        if not args.no_render:
            env.render()

        for step in range(1000):
            actions = env.action_space.sample()
            observations, rewards, terminated, truncated, info = env.step(actions)
            print(f"Step {step + 1}: Actions={actions}, Rewards={rewards}")

            if not args.no_render:
                env.render()
                time.sleep(0.05)

            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
