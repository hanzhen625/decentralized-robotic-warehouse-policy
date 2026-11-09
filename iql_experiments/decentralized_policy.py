
from collections import deque

from rware.warehouse import Warehouse, RewardType, ObservationType


# ==========================================
# 1. CREATE WAREHOUSE
# ==========================================

env = Warehouse(
    shelf_rows=1,
    shelf_columns=3,
    column_height=8,
    n_agents=2,
    msg_bits=0,
    sensor_range=1,
    request_queue_size=2,
    max_inactivity_steps=None,
    max_steps=500,
    reward_type=RewardType.INDIVIDUAL,
    observation_type=ObservationType.FLATTENED,
    render_mode=None
)

observations, info = env.reset(seed=42)

robot = env.agents[0]

print("Warehouse initialized!")
print("Grid size:", env.grid_size)
print("Delivery goals:", env.goals)
print("Robot position:", (robot.x, robot.y))
print("Action space:", env.action_space)


# ==========================================
# 2. ACTIONS AND DIRECTIONS
# ==========================================

NOOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
TOGGLE_LOAD = 4

# RWARE direction values:
# UP=0, DOWN=1, LEFT=2, RIGHT=3

DIRECTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0)
}

CLOCKWISE = [0, 3, 1, 2]


def get_position(agent):
    return int(agent.x), int(agent.y)


def get_direction(agent):
    return agent.dir.value


def turn_action(current, desired):
    current_index = CLOCKWISE.index(current)
    desired_index = CLOCKWISE.index(desired)

    difference = (desired_index - current_index) % 4

    if difference == 0:
        return FORWARD
    elif difference == 1:
        return RIGHT
    elif difference == 3:
        return LEFT
    else:
        return RIGHT


# ==========================================
# 3. BFS PATHFINDING
# ==========================================

def find_path(start, goal, blocked):

    # RWARE grid_size = (height, width)
    height, width = env.grid_size

    queue = deque([start])
    previous = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for dx, dy in DIRECTION_DELTAS.values():

            nx = current[0] + dx
            ny = current[1] + dy

            neighbor = (nx, ny)

            # Correct coordinate boundaries
            if not (0 <= nx < width and 0 <= ny < height):
                continue

            if neighbor in blocked:
                continue

            if neighbor in previous:
                continue

            previous[neighbor] = current
            queue.append(neighbor)

    if goal not in previous:
        return None

    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = previous[current]

    return list(reversed(path))


# ==========================================
# 4. OBSTACLE DETECTION
# ==========================================

def get_blocked_cells(agent, target):

    blocked = set()

    # Other robots
    for other in env.agents:
        if other is not agent:
            blocked.add(get_position(other))

    # Other shelves block movement while carrying
    if agent.carrying_shelf is not None:
        for shelf in env.shelfs:
            if shelf is not agent.carrying_shelf:
                blocked.add(
                    (int(shelf.x), int(shelf.y))
                )

    return blocked


# ==========================================
# 5. EXECUTE ACTION
# ==========================================

def execute_action(action):

    observations, rewards, terminated, truncated, info = (
        env.step([action, NOOP])
    )

    episode_done = (
        any(terminated)
        if isinstance(terminated, (list, tuple))
        else bool(terminated)
    )

    episode_truncated = (
        any(truncated)
        if isinstance(truncated, (list, tuple))
        else bool(truncated)
    )

    return rewards, episode_done, episode_truncated


# ==========================================
# 6. NAVIGATION
# ==========================================

def navigate(agent, target, max_steps=150):

    print("\nNavigating to:", target)

    for step in range(max_steps):

        position = get_position(agent)

        if position == target:
            print("Reached target!")
            return True, False

        blocked = get_blocked_cells(agent, target)

        path = find_path(position, target, blocked)

        if path is None or len(path) < 2:
            print("No available path.")
            return False, False

        next_position = path[1]

        dx = next_position[0] - position[0]
        dy = next_position[1] - position[1]

        if dx == 1:
            desired_direction = 3
        elif dx == -1:
            desired_direction = 2
        elif dy == 1:
            desired_direction = 1
        else:
            desired_direction = 0

        action = turn_action(
            get_direction(agent),
            desired_direction
        )

        old_position = get_position(agent)
        old_direction = get_direction(agent)

        rewards, terminated, truncated = execute_action(action)

        new_position = get_position(agent)
        new_direction = get_direction(agent)

        print(
            f"Step {step + 1}: "
            f"{old_position} -> {new_position}, "
            f"Direction={agent.dir.name}, "
            f"Action={action}, "
            f"Rewards={rewards}"
        )

        # Detect completed delivery
        if any(float(r) > 0 for r in rewards):
            print("Positive reward received!")

            if agent.carrying_shelf is not None:
                print("Shelf delivery reward detected!")
                return True, True

        if (
            old_position == new_position
            and old_direction == new_direction
        ):
            print("Action had no effect.")
            return False, False

        if terminated or truncated:
            print("Episode ended.")
            return new_position == target, False

    if get_position(agent) == target:
        print("Reached target!")
        return True, False

    print("Navigation step limit reached.")
    return False, False


# ==========================================
# 7. SELECT REQUESTED SHELF
# ==========================================

print("\nRobot starting position:", get_position(robot))
print("Robot starting direction:", robot.dir.name)

target_shelf = env.request_queue[0]

shelf_target = (
    int(target_shelf.x),
    int(target_shelf.y)
)

print("Requested shelf:", shelf_target)


# ==========================================
# 8. PICK UP SHELF
# ==========================================

reached_shelf, _ = navigate(robot, shelf_target)

if reached_shelf:

    print("\nAttempting shelf pickup...")

    rewards, terminated, truncated = execute_action(TOGGLE_LOAD)

    if robot.carrying_shelf is not None:
        print("SUCCESS! Robot picked up the shelf!")
    else:
        print("Shelf pickup failed.")

else:
    print("Shelf pickup skipped.")


# ==========================================
# 9. DELIVER SHELF
# ==========================================

if robot.carrying_shelf is not None:

    print("\nStarting delivery...")

    goal = env.goals[0]

    goal_target = (
        int(goal[0]),
        int(goal[1])
    )

    print("Delivery goal:", goal_target)

    reached_goal, delivered = navigate(
        robot,
        goal_target
    )

    if delivered:
        print("\nSUCCESS! Shelf delivery rewarded!")

    elif reached_goal:
        print("\nRobot reached the delivery station.")
        print("No positive delivery reward detected yet.")

    else:
        print("\nRobot could not reach the delivery station.")

else:
    print("Delivery skipped: Robot is not carrying a shelf.")


# ==========================================
# 10. CLEAN UP
# ==========================================

env.close()

print("\nSimulation finished.")
