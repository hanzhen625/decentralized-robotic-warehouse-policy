"""Shared training/simulation setup and configurable delivery rewards."""

from collections import deque
from functools import lru_cache

import gymnasium as gym
import numpy as np

import rware  # noqa: F401 - register environments
from rware.warehouse import RewardType


GAMMA = 0.99  # Must match PPO's discount for potential-based shaping.
DELIVERY_REWARD = 1.0
RETURN_REWARD = 0.5
EXPLORE_REWARD = 0.002
STEP_COST = 0.001
SHAPING_SCALE = 0.5
DISTANCE_SCALE = 20.0
REWARD_SETTINGS = dict(delivery=DELIVERY_REWARD, step_cost=STEP_COST, gamma=GAMMA,
                       return_reward=RETURN_REWARD, exploration=EXPLORE_REWARD,
                       shaping_scale=SHAPING_SCALE, distance_scale=DISTANCE_SCALE)


@lru_cache(maxsize=128)
def distances(shape, blocked, targets):
    """Shortest grid distances; shelves obstruct loaded robots, traffic is ignored.

    Cache maps until shelf occupancy changes. This supplies a distance heuristic,
    not navigation actions or centrally assigned tasks.
    """
    result = {p: 0 for p in targets}
    queue = deque(targets)
    while queue:
        x, y = queue.popleft()
        for p in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= p[0] < shape[1] and 0 <= p[1] < shape[0]
                    and p not in blocked and p not in result):
                result[p] = result[x, y] + 1
                queue.append(p)
    return result


def distance(field, position):
    """A newly picked-up shelf may start on a blocked storage cell."""
    x, y = position
    return field.get(position, min(
        (field.get(p, 100) + 1 for p in
         ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))), default=100))


class DeliveryRewards(gym.Wrapper):
    """Broadcast destination features; optionally shape each robot's own reward."""

    def __init__(self, env, reward_system):
        super().__init__(env)
        self.reward_system = reward_system
        # Fixed request slots (x offset, y offset, active), stations, task flags,
        # current destination offset, and estimated remaining delivery distance.
        extra = 3 * env.unwrapped.request_queue_size + 2 * len(env.unwrapped.goals) + 5
        self.observation_space = gym.spaces.Tuple(tuple(
            gym.spaces.Box(np.r_[s.low, np.full(extra, -1)],
                           np.r_[s.high, np.ones(extra)], dtype=np.float32)
            for s in env.observation_space))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.storage = {(s.x, s.y) for s in self.unwrapped.shelfs}
        obs, self.potential = self.features(obs)
        return obs, info

    def features(self, observations):
        world = self.unwrapped
        height, width = world.grid_size
        scale = np.array([width - 1, height - 1], dtype=np.float32)
        carried = {a.carrying_shelf for a in world.agents if a.carrying_shelf}
        blocked = frozenset((s.x, s.y) for s in world.shelfs if s not in carried)
        occupied = {(s.x, s.y) for s in world.shelfs}
        goals = tuple(world.goals)  # Warehouse stores goals as (x, y).
        goal_field = distances(world.grid_size, blocked, goals)
        result, potentials = [], []
        for agent, obs in zip(world.agents, observations):
            pos = (agent.x, agent.y)
            # A request already carried by another robot is marked unavailable.
            requests = [s for s in sorted(world.request_queue, key=lambda s: s.id)
                        if s not in carried or s is agent.carrying_shelf]
            choices = [(abs(s.x - agent.x) + abs(s.y - agent.y) + 1
                        + distance(goal_field, (s.x, s.y)), (s.x, s.y))
                       for s in requests]
            remaining, target = min(choices, default=(0, pos))
            requested = agent.carrying_shelf in world.request_queue
            returning = bool(agent.carrying_shelf and not requested)
            if requested:
                target = min(goals, key=lambda p: abs(p[0] - agent.x) + abs(p[1] - agent.y))
                remaining = distance(goal_field, pos)
            elif returning:
                vacant = self.storage - occupied
                if pos in self.storage:
                    vacant.add(pos)  # It may unload the shelf it currently carries.
                target = min(vacant, key=lambda p: abs(p[0] - agent.x) + abs(p[1] - agent.y),
                             default=pos)
                field = distances(world.grid_size, blocked, tuple(sorted(vacant)))
                remaining += distance(field, pos) + 1  # Travel, unload, then fetch.
            slots = [v for s in requests for v in (*((np.array([s.x, s.y]) - pos) / scale), 1)]
            slots += [0] * (3 * world.request_queue_size - len(slots))
            stations = (np.asarray(goals) / scale).ravel()
            task = [int(requested), int(returning), *((np.array(target) - pos) / scale),
                    min(remaining / 100, 1)]
            result.append(np.concatenate([obs, slots, stations, task]).astype(np.float32))
            # Positive, bounded potential: waiting earns no positive shaping bonus.
            potentials.append(SHAPING_SCALE * np.exp(-remaining / DISTANCE_SCALE)
                              if world.request_queue else 0.0)
        return tuple(result), np.asarray(potentials)

    def step(self, actions):
        obs, base, terminated, truncated, info = self.env.step(actions)
        obs, potential = self.features(obs)
        # At a real terminal state potential is zero. At a time limit it remains
        # nonzero: RobotVecEnv bootstraps that truncated episode's value.
        shaping = GAMMA * potential - self.potential
        if self.reward_system == "delivery":
            shaping = np.zeros_like(shaping)
        self.potential = potential
        rewards = DELIVERY_REWARD * np.asarray(base) - STEP_COST + shaping
        info.update(base_rewards=list(base), shaping_rewards=shaping.tolist())
        return obs, rewards.tolist(), terminated, truncated, info


def make_env(reward_system="local_return", render_mode=None):
    """Keep the physical task identical across all three reward systems."""
    if reward_system not in ("baseline", "delivery", "shaped", "local_return"):
        raise ValueError(f"Unknown reward system: {reward_system}")
    env = gym.make(
        "rware-tiny-2ag-v2", shelf_rows=1, shelf_columns=3, column_height=8,
        request_queue_size=2, replace_requests=False, max_steps=1000,
        require_shelf_return=reward_system == "local_return",
        sensor_range=1, msg_bits=0, normalised_coordinates=True,
        reward_type=(RewardType.TWO_STAGE if reward_system == "baseline"
                     else RewardType.INDIVIDUAL), render_mode=render_mode,
    )
    if reward_system == "local_return":
        from local_return_rewards import LocalReturnRewards
        return LocalReturnRewards(env)
    return env if reward_system == "baseline" else DeliveryRewards(env, reward_system)
