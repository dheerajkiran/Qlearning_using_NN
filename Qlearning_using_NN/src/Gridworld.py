import random
import numpy as np


class Gridworld(object):
    """
    Classic 3x4 grid with a wall, a goal, and a penalty. States are normalized to [0,1]^2.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left.
    Slip: 10% to the "right" of intended direction, 10% to the "left".
    """
    def __init__(self, goal=1, penalty=-1):
        self.n_rows = 3
        self.n_columns = 4

        self.start_state = np.array([0, 0])
        self.wall_state = np.array([1, 1])
        self.goal_state = np.array([2, 3])
        self.penalty_state = np.array([1, 3])

        self.goal_reward = goal
        self.penalty_reward = penalty

        # Initialize current state
        self.state = self.start_state

    def step(self, action):
        """
        Take one step with slip dynamics and return (normalized_next_state, reward, done).
        """
        # Slip: 80% intended, 10% rotate right, 10% rotate left
        u = random.random()
        if u < 0.1:
            action = (action + 1) % 4
        elif u < 0.2:
            action = (action + 3) % 4

        row, col = self.state

        # Propose next state
        if action == 0:         # Up
            next_state = np.array([min(row + 1, self.n_rows - 1), col])
        elif action == 1:       # Right
            next_state = np.array([row, min(col + 1, self.n_columns - 1)])
        elif action == 2:       # Down
            next_state = np.array([max(row - 1, 0), col])
        else:                   # Left
            next_state = np.array([row, max(col - 1, 0)])

        # Bounce off the wall
        if np.array_equal(next_state, self.wall_state):
            next_state = self.state

        # Terminal checks
        if np.array_equal(next_state, self.goal_state):
            reward = self.goal_reward
            done = True
        elif np.array_equal(next_state, self.penalty_state):
            reward = self.penalty_reward
            done = True
        else:
            reward = -0.04  # small step cost
            done = False

        self.state = next_state

        # Normalize [row, col] into [0,1] range so the network has a nice input scale
        normalized_next = next_state / np.array([self.n_rows - 1, self.n_columns - 1], dtype=float)

        return normalized_next, reward, done

    def reset(self):
        """
        Reset to the start square (bottom-left). Returns raw state (not normalized).
        """
        self.state = self.start_state
        return self.state
