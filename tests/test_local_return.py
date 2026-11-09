"""Exact shelf identity/home restoration and absence of remote pickup knowledge."""

import numpy as np

from rware.warehouse import Action
from warehouse_rewards import RETURN_REWARD, make_env


def local_observations(env):
    return tuple(env.unwrapped._make_obs(a) for a in env.unwrapped.agents)


def test_delivery_is_not_terminal_but_home_return_is():
    env = make_env("local_return")
    try:
        env.reset(seed=42)
        world = env.unwrapped
        shelf = world.request_queue[0]
        world.request_queue, world.initial_request_ids = [shelf], {shelf.id}
        world.request_queue_size = 1
        agent = world.agents[0]
        agent.x, agent.y = shelf.x, shelf.y = world.goals[0]
        agent.carrying_shelf = shelf
        world.agents[1].x, world.agents[1].y = 0, 0
        env.payload[0] = (shelf.id, True)
        world._recalc_grid()
        _, env.potential, _ = env.features(local_observations(env))
        obs, _, done, truncated, info = env.step([0, 0])
        assert not done and not truncated and not world.request_queue
        assert info["holes"] == info["shelves_away"] == 1
        assert env.observation_space.contains(obs)
        agent.x, agent.y = shelf.x, shelf.y = world.shelf_origins[shelf.id]
        world._recalc_grid()
        obs, _, done, truncated, info = env.step([Action.TOGGLE_LOAD.value, 0])
        assert done and not truncated and info["task_complete"]
        assert info["return_rewards"] == [RETURN_REWARD, 0]
        assert info["holes"] == info["shelves_away"] == 0
        np.testing.assert_array_equal(env.potential, [0, 0])
        # Re-picking and dropping at home cannot earn another return bonus.
        env.step([Action.TOGGLE_LOAD.value, 0])
        _, _, _, _, info = env.step([Action.TOGGLE_LOAD.value, 0])
        assert info["return_rewards"] == [0, 0]
    finally:
        env.close()


def test_remote_pickup_and_delivery_do_not_change_other_robots_input():
    env = make_env("local_return")
    try:
        env.reset(seed=42)
        world = env.unwrapped
        observer, other = world.agents
        observer.x, observer.y = 0, 0
        shelf = max(world.request_queue, key=lambda s: s.x + s.y)
        # Simulate a private earlier sighting that is now outside sensor range.
        env.memory[0][shelf.id] = ((shelf.x, shelf.y), True)
        world._recalc_grid()
        before, _, _ = env.features(local_observations(env))
        other.carrying_shelf = shelf
        other.x, other.y = shelf.x, shelf.y = world.goals[0]
        world._recalc_grid()
        picked_up, _, _ = env.features(local_observations(env))
        np.testing.assert_array_equal(before[0], picked_up[0])
        assert shelf.id in env.memory[0]  # No remote availability notification.
        world.request_queue.remove(shelf)  # Remote delivery is also not broadcast.
        delivered, _, _ = env.features(local_observations(env))
        np.testing.assert_array_equal(before[0], delivered[0])
        observer.x, observer.y = env.homes[shelf.id]
        world._recalc_grid()
        env.features(local_observations(env))
        assert shelf.id not in env.memory[0]  # It finally observes the empty home.
    finally:
        env.close()


def test_timeout_after_deliveries_still_bootstraps_return_work():
    from train_decentralized import RobotVecEnv

    env = RobotVecEnv("local_return")
    try:
        env.reset()
        world = env.env.unwrapped
        world.request_queue.clear()
        shelf = world.shelfs[0]
        agent = world.agents[0]
        agent.carrying_shelf = shelf
        agent.x, agent.y = shelf.x, shelf.y = world.goals[0]
        world.agents[1].x, world.agents[1].y = 0, 0
        world.max_steps = 1
        world._recalc_grid()
        env.env.payload[0] = (shelf.id, False)
        _, env.env.potential, _ = env.env.features(local_observations(env.env))
        _, _, dones, infos = env.step(np.zeros(2, dtype=int))
        assert dones.all()
        assert all(info["TimeLimit.truncated"] and not info["is_success"] for info in infos)
        assert len(world.request_queue) == 2  # Adapter automatically reset.
    finally:
        env.close()


def test_filled_holes_with_swapped_shelves_are_not_success():
    env = make_env("local_return")
    try:
        env.reset(seed=42)
        world = env.unwrapped
        world.request_queue.clear()
        first, second = world.shelfs[:2]
        first.x, second.x = second.x, first.x
        first.y, second.y = second.y, first.y
        metrics = world.task_metrics()
        assert metrics["holes"] == 0
        assert metrics["shelves_away"] == 2
        assert not metrics["task_complete"]
        world.max_steps = 1
        _, _, done, truncated, _ = env.step([0, 0])
        assert not done and truncated  # Time limit remains distinct from success.
    finally:
        env.close()
