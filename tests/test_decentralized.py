"""Check the multi-agent adapter's observation boundaries and episode endings."""

import numpy as np
import pytest

from train_decentralized import RobotVecEnv


@pytest.mark.parametrize("success", [False, True])
def test_joint_step_and_episode_end(success):
    env = RobotVecEnv("baseline")
    try:
        env.seed(42)
        observations = env.reset()
        assert observations.shape == (2, *env.observation_space.shape)
        for i, agent in enumerate(env.env.unwrapped.agents):
            np.testing.assert_array_equal(observations[i], env.env.unwrapped._make_obs(agent))

        env.step(np.zeros(2, dtype=int))
        assert env.env.unwrapped._cur_steps == 1  # One joint step, not one per robot.
        env.env.unwrapped.max_steps = 2
        if success:
            env.env.unwrapped.request_queue.clear()
        observations, rewards, dones, infos = env.step(np.zeros(2, dtype=int))
        assert dones.all()
        assert rewards.shape == (2,)
        assert env.env.unwrapped._cur_steps == 0  # Shared episode automatically reset.
        for info in infos:
            assert info["TimeLimit.truncated"] == (not success)
            assert info["terminal_observation"].shape == env.observation_space.shape
            assert info["episode"]["l"] == 2
        assert len(env.env.unwrapped.request_queue) == 2
    finally:
        env.close()
