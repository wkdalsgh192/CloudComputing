import numpy as np
import random

class MazeEnvironment:
    def __init__(self, N, M, wall_prob=0.1):
        self.N, self.M = N, M
        self.grid = np.zeros((N, M), dtype=int)  # 0: empty, 1: wall, 2: goal
        
        self.start = (0, 0)
        self.goal = (N - 1, M - 1)
        self.grid[self.goal] = 2

        # --- Randomly place walls (10% of cells by default) ---
        for r in range(N):
            for c in range(M):
                if (r, c) not in [self.start, self.goal] and random.random() < wall_prob:
                    self.grid[r, c] = 1  # wall

        # --- Define possible actions ---
        self.actions = ["up", "down", "left", "right"]

    def reset(self):
        """Reset to start position."""
        return self.start

    def step(self, state, action):
        """Takes an action and returns (next_state, reward, done)."""
        r, c = state

        if action == "right":
            c += 1
        elif action == "left":
            c -= 1
        elif action == "up":
            r -= 1
        elif action == "down":
            r += 1

        # --- Check boundaries ---
        if r < 0 or r >= self.N or c < 0 or c >= self.M:
            return state, -10, False  # penalty for hitting wall (outside bounds)

        # --- Check for walls ---
        if self.grid[r, c] == 1:
            return state, -10, False  # penalty for hitting a wall

        next_state = (r, c)

        # --- Check if goal reached ---
        if next_state == self.goal:
            return next_state, 100, True  # big reward for reaching goal

        # --- Normal step ---
        return next_state, -1, False  # small penalty for each step


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.random.uniform(-0.1, 0.1, (env.N * env.M, 4))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.9
    def choose_action(self, state, episode):
        # Implement epsilon-greedy with custom exploration rule
        if random() < self.epsilon:
            # exploitation
        else:
            #exploration

    def update(self, state, action, reward, next_state):
        # Implement Q-Table update
        pass