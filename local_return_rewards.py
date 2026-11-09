"""Local sensing and private robot memory for delivery-and-return episodes."""

import gymnasium as gym
import numpy as np

from warehouse_rewards import (GAMMA, STEP_COST, DELIVERY_REWARD, RETURN_REWARD,
                               SHAPING_SCALE, DISTANCE_SCALE, EXPLORE_REWARD,
                               distances, distance)


class LocalReturnRewards(gym.Wrapper):
    """No live request board, remote shelf coordinates, or other-robot status.

    A static warehouse map and station locations are known. Each robot keeps
    its own last-seen shelf records. Target hints come only from that memory;
    navigation and load/unload actions are still learned by PPO.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple(tuple(
            gym.spaces.Box(np.r_[s.low, np.full(14, -1)],
                           np.r_[s.high, np.ones(14)], dtype=np.float32)
            for s in env.observation_space))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        world = self.unwrapped
        self.homes = dict(world.shelf_origins)  # Static inventory, not live positions.
        self.storage = frozenset(self.homes.values())
        self.memory = [dict() for _ in world.agents]
        self.seen = [set() for _ in world.agents]
        self.ever_seen = [set() for _ in world.agents]
        self.payload = [None for _ in world.agents]
        self.own_delivered = [set() for _ in world.agents]
        self.delivered_ids, self.rewarded_returns = set(), set()  # Scoring only.
        obs, self.potential, _ = self.features(obs)
        return obs, info

    def features(self, observations):
        world = self.unwrapped
        scale = np.array([world.grid_size[1] - 1, world.grid_size[0] - 1])
        goals = tuple(world.goals)
        result, potentials, discoveries = [], [], []
        for i, (agent, obs) in enumerate(zip(world.agents, observations)):
            pos = (agent.x, agent.y)
            visible = lambda p: (abs(p[0] - agent.x) <= world.sensor_range
                                 and abs(p[1] - agent.y) <= world.sensor_range)
            # Only read shelf locations inside this robot's actual sensor square.
            local = {s.id: s for s in world.shelfs if visible((s.x, s.y))}
            for sid, (location, requested) in list(self.memory[i].items()):
                if visible(location) and (sid not in local or
                                          (local[sid].x, local[sid].y) != location):
                    del self.memory[i][sid]  # Learn about a hole only by seeing it.
            for sid, shelf in local.items():
                if shelf is not agent.carrying_shelf:
                    # Request flags are the same flags visible in RWARE's local sensors.
                    self.memory[i][sid] = ((shelf.x, shelf.y), shelf in world.request_queue)
            visible_storage = {p for p in self.storage if visible(p)}
            discoveries.append(len(visible_storage - self.ever_seen[i]))
            self.ever_seen[i].update(visible_storage)
            self.seen[i].update(visible_storage)
            payload = self.payload[i]
            searching, delivering, returning = payload is None, False, False
            home = pos
            # No filtering based on another robot's pickup, carrying, or delivery.
            known = [(sid, p, req) for sid, (p, req) in self.memory[i].items()
                     if (req and sid not in self.own_delivered[i]) or p != self.homes[sid]]
            if payload:
                sid, delivering = payload
                home = self.homes[sid]  # The carried shelf's own home metadata.
                returning = not delivering
                target = (min(goals, key=lambda p: abs(p[0] - agent.x) + abs(p[1] - agent.y))
                          if delivering else home)
            elif known:
                _, target, _ = min(known, key=lambda item:
                                  abs(item[1][0] - agent.x) + abs(item[1][1] - agent.y))
            else:
                # Independent exploration hint. No exclusive task allocation.
                target = min(self.storage, key=lambda p:
                             (p in self.seen[i], abs(p[0] - agent.x) + abs(p[1] - agent.y), p))
                if self.seen[i] >= self.storage:
                    self.seen[i].clear()  # Revisit stale shelves; novelty reward stays capped.
            blocked = self.storage - {home} if payload else frozenset()
            field = distances(world.grid_size, frozenset(blocked), (target,))
            remaining = distance(field, pos)
            # Loaded delivery also needs a return journey and an unload action.
            if delivering:
                remaining += abs(target[0] - home[0]) + abs(target[1] - home[1]) + 1
            elif searching and known:
                remaining += 2  # Pickup and eventual unload are still required.
            else:
                remaining += int(returning)
            extras = [int(searching), int(delivering), int(returning),
                      *((np.array(target) - pos) / scale),
                      *((np.array(home) - pos) / scale), min(remaining / 100, 1),
                      min(len(known) / 10, 1), int(bool(known))]
            stations = (np.asarray(goals) / scale).ravel()
            result.append(np.r_[obs, extras, stations].astype(np.float32))
            potentials.append(SHAPING_SCALE * np.exp(-remaining / DISTANCE_SCALE))
        return tuple(result), np.asarray(potentials), np.asarray(discoveries)

    def step(self, actions):
        world = self.unwrapped
        before = [a.carrying_shelf for a in world.agents]
        obs, base, terminated, truncated, info = self.env.step(actions)
        returns = np.zeros(world.n_agents)
        for i, agent in enumerate(world.agents):
            shelf = agent.carrying_shelf
            if before[i] is None and shelf is not None:
                requested = self.memory[i].get(shelf.id, (None, False))[1]
                self.payload[i] = (shelf.id, bool(requested))
            if base[i] > 0 and shelf is not None:
                self.payload[i] = (shelf.id, False)  # Its own delivery receipt only.
                self.own_delivered[i].add(shelf.id)
                self.delivered_ids.add(shelf.id)
            if before[i] is not None and shelf is None:
                old = before[i]
                if ((old.x, old.y) == self.homes[old.id] and old.id in self.delivered_ids
                        and old.id not in self.rewarded_returns):
                    returns[i] = RETURN_REWARD
                    self.rewarded_returns.add(old.id)
                self.payload[i] = None
        obs, potential, discoveries = self.features(obs)
        if terminated:
            potential[:] = 0  # All deliveries AND all shelf identities restored.
        shaping = GAMMA * potential - self.potential
        self.potential = potential
        rewards = DELIVERY_REWARD * np.asarray(base) + returns - STEP_COST + shaping
        rewards += EXPLORE_REWARD * discoveries
        info.update(base_rewards=list(base), return_rewards=returns.tolist(),
                    shaping_rewards=shaping.tolist())
        return obs, rewards.tolist(), terminated, truncated, info
