"""Warehouse environments for CS F407 RL laboratory."""
from dataclasses import dataclass
from typing import Tuple

GRID_SIZE = 5
OBSTACLES = {(0, 2), (1, 2), (3, 2), (4, 2)}
START = (2, 0)
GOAL = (2, 4)
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ACTION_NAMES = ("up", "down", "left", "right", "wait")
GAMMA = 0.95
MAX_STEPS = 40


def cell_index(pos: Tuple[int, int]) -> int:
    return 5 * pos[0] + pos[1]


def one_hot(pos: Tuple[int, int]):
    import numpy as np
    x = np.zeros(25, dtype=np.float32)
    x[cell_index(pos)] = 1.0
    return x


class SingleRobotEnv:
    def __init__(self):
        self.position = START
        self.steps = 0

    def reset(self, position=START):
        if not self._free(position):
            raise ValueError(f"Invalid starting position: {position}")
        self.position = tuple(position)
        self.steps = 0
        return self.position

    @staticmethod
    def _free(pos):
        r, c = pos
        return 0 <= r < 5 and 0 <= c < 5 and pos not in OBSTACLES

    def step(self, action):
        if not 0 <= int(action) < 5:
            raise ValueError("Action must be in [0, 4]")
        old = self.position
        dr, dc = ACTIONS[int(action)]
        proposed = (old[0] + dr, old[1] + dc)
        invalid = not self._free(proposed)
        new = old if invalid else proposed
        self.position = new
        self.steps += 1
        terminated = new == GOAL
        reward = 10.0 if terminated else (-1.0 if invalid else -0.1)
        truncated = self.steps >= MAX_STEPS and not terminated
        return new, reward, terminated, truncated, {"invalid": int(invalid)}


@dataclass
class JointStepInfo:
    collision: bool
    invalid_moves: int
    newly_arrived: int


class TwoRobotEnv:
    START_A, START_B = (2, 0), (2, 4)
    GOAL_A, GOAL_B = (2, 4), (2, 0)

    def __init__(self):
        self.positions = (self.START_A, self.START_B)
        self.arrived = (False, False)
        self.steps = 0

    def reset(self, positions=None):
        if positions is None:
            positions = (self.START_A, self.START_B)
        if any(not SingleRobotEnv._free(p) for p in positions):
            raise ValueError("Both starts must be free cells")
        if positions[0] == positions[1]:
            raise ValueError("Robots cannot start in the same cell")
        self.positions = (tuple(positions[0]), tuple(positions[1]))
        self.arrived = (self.positions[0] == self.GOAL_A,
                        self.positions[1] == self.GOAL_B)
        self.steps = 0
        return self.positions

    @staticmethod
    def _decode(joint_action):
        return divmod(int(joint_action), 5)

    def step(self, joint_action):
        if not 0 <= int(joint_action) < 25:
            raise ValueError("Joint action must be in [0, 24]")
        a_a, a_b = self._decode(joint_action)
        old_a, old_b = self.positions
        old_arrived = self.arrived
        proposed, invalid = [], 0
        for old, action, arrived in ((old_a, a_a, old_arrived[0]),
                                     (old_b, a_b, old_arrived[1])):
            if arrived:
                proposed.append(old)
                continue
            dr, dc = ACTIONS[action]
            candidate = (old[0] + dr, old[1] + dc)
            bad = not SingleRobotEnv._free(candidate)
            invalid += int(bad)
            proposed.append(old if bad else candidate)
        new_a, new_b = proposed
        same_cell = new_a == new_b
        swap = new_a == old_b and new_b == old_a
        collision = same_cell or swap
        if collision:
            # Keep old positions for reporting, and end the episode.
            self.steps += 1
            self.positions = (old_a, old_b)
            reward = -10.0
            terminated = True
            truncated = False
            info = JointStepInfo(True, invalid, 0)
            return self.positions, reward, terminated, truncated, info.__dict__

        new_arrived = (old_arrived[0] or new_a == self.GOAL_A,
                       old_arrived[1] or new_b == self.GOAL_B)
        newly = int(not old_arrived[0] and new_arrived[0]) + int(not old_arrived[1] and new_arrived[1])
        both = int(new_arrived[0] and new_arrived[1])
        # Invalid moves by robots already at their goals are excluded above.
        reward = -0.1 - invalid + 5.0 * newly + 5.0 * both
        self.positions, self.arrived = (new_a, new_b), new_arrived
        self.steps += 1
        terminated = both == 1
        truncated = self.steps >= MAX_STEPS and not terminated
        info = JointStepInfo(False, invalid, newly)
        return self.positions, reward, terminated, truncated, info.__dict__


def joint_one_hot(positions):
    import numpy as np
    return np.concatenate([one_hot(positions[0]), one_hot(positions[1])])
