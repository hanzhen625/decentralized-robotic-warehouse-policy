"""Verify dense rewards reflect progress, not idling or repeated loading."""

import numpy as np
import pytest

from rware.warehouse import Action, Direction
from warehouse_rewards import GAMMA, STEP_COST, make_env


@pytest.mark.parametrize("mode", ["delivery", "shaped"])
def test_features_and_idle_cost(mode):
    env = make_env(mode)
    try:
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        before = env.potential.copy()
        obs, rewards, _, _, info = env.step([Action.NOOP.value] * 2)
        assert env.observation_space.contains(obs)
        expected = (-STEP_COST + (GAMMA - 1) * before if mode == "shaped"
                    else np.full(2, -STEP_COST))
        np.testing.assert_allclose(rewards, expected)
        assert max(rewards) < 0  # Waiting earns no positive bonus.
        assert info["base_rewards"] == [0, 0]
    finally:
        env.close()


def test_delivery_and_terminal_potential():
    env = make_env("shaped")
    try:
        env.reset(seed=42)
        world = env.unwrapped
        shelf = world.request_queue[0]
        world.request_queue = [shelf]  # One remaining request, padded feature slot.
        agent = world.agents[0]
        agent.x, agent.y = shelf.x, shelf.y = world.goals[0]
        agent.carrying_shelf = shelf
        world.agents[1].x, world.agents[1].y = 0, 0
        world._recalc_grid()
        _, env.potential = env.features(tuple(world._make_obs(a) for a in world.agents))
        before = env.potential.copy()
        obs, rewards, done, _, info = env.step([Action.NOOP.value] * 2)
        assert done and not world.request_queue
        assert env.observation_space.contains(obs)
        assert info["base_rewards"] == [1, 0]
        np.testing.assert_array_equal(env.potential, [0, 0])
        np.testing.assert_allclose(rewards, np.array([1, 0]) - STEP_COST - before)
    finally:
        env.close()


def test_forward_progress_and_time_limit():
    env = make_env("shaped")
    try:
        env.reset(seed=42)
        world = env.unwrapped
        agent = world.agents[0]
        shelf = world.request_queue[0]
        # An empty robot can travel beneath shelves. Choose one horizontal step
        # toward the requested shelf, away from the other robot.
        world.request_queue = [shelf]
        agent.x, agent.y = (shelf.x - 1 if shelf.x else 1), shelf.y
        agent.dir = Direction.RIGHT if agent.x < shelf.x else Direction.LEFT
        world.agents[1].x, world.agents[1].y = 0, 0
        world.max_steps = 1
        world._recalc_grid()
        _, env.potential = env.features(tuple(world._make_obs(a) for a in world.agents))
        obs, rewards, done, _, _ = env.step([Action.FORWARD.value, Action.NOOP.value])
        assert done and world.request_queue  # Time limit, not task completion.
        assert env.potential[0] > 0  # Preserve potential for timeout bootstrapping.
        assert rewards[0] > 0
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_repeated_pickup_does_not_farm_reward():
    env = make_env("shaped")
    try:
        env.reset(seed=42)
        world = env.unwrapped
        shelf = world.request_queue[0]
        world.agents[0].x, world.agents[0].y = shelf.x, shelf.y
        world.agents[1].x, world.agents[1].y = 0, 0
        world._recalc_grid()
        _, env.potential = env.features(tuple(world._make_obs(a) for a in world.agents))
        before = env.potential.copy()
        _, pickup, _, _, _ = env.step([Action.TOGGLE_LOAD.value, Action.NOOP.value])
        _, drop, _, _, _ = env.step([Action.TOGGLE_LOAD.value, Action.NOOP.value])
        np.testing.assert_allclose(env.potential, before)  # Same physical state.
        assert pickup[0] + GAMMA * drop[0] < 0
    finally:
        env.close()
