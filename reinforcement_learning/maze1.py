import numpy as np
import random
import matplotlib.pyplot as plt


class MazeEnvironment:
    def __init__(self, N, M):
        self.N, self.M = N, M

        self.grid = np.zeros((N, M)) # 0: empty, 1: wall, 2: goal
        self.start = (0, 0)
        self.goal = (N-1, M-1)
        self.grid[N-1, M-1] = 2

        self.state_space_size = N*M
        self.action_space_size = 4

        # Randomly place walls (10% of cells)
        num_walls = int(0.1 * self.state_space_size)
        walls_placed = 0
        while walls_placed < num_walls:
            i = random.randint(0, N-1)
            j = random.randint(0, M-1)
            if (i, j) != self.start and (i, j) != self.goal and self.grid[i, j] == 0:
                self.grid[i, j] = 1
                walls_placed += 1

        self.action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def step(self, state, action):
        row, col = state
        d_row, d_col = self.action_map[action]
        new_row, new_col = row + d_row, col + d_col

        done = False

        if new_row < 0 or new_row >= self.N or new_col < 0 or new_col >= self.M or self.grid[new_row, new_col] == 1:
            next_state = state
            reward = -10
        
        elif (new_row, new_col) == self.goal:
            next_state = (new_row, new_col)
            reward = 100
            done = True
        
        else:
            next_state = (new_row, new_col)
            reward = -1
        
        return next_state, reward, done
    
    def state_to_index(self, state):
        row, col = state
        return row * self.M + col

    def reset(self):
        return self.start
    

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.random.uniform(-0.1, 0.1, (env.state_space_size, env.action_space_size))

        self.alpha = 0.1
        self.gamma = 0.9

        self.initial_epsilon = 0.9
        self.final_epsilon = 0.1
        self.epsilon_decay_episodes = 500
        self.epsilon = self.initial_epsilon

        self.random_action_counter = 0
    
    def choose_action(self, state, episode):
        state_idx = self.env.state_to_index(state)

        if episode > 0 and episode % 5 == 0 and self.random_action_counter < 3:
            self.random_action_counter += 1
            return random.randint(0, self.env.action_space_size - 1)

        if episode % 5 != 0:
            self.random_action_counter = 0
        
        if episode < self.epsilon_decay_episodes:
            self.epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (episode / self.epsilon_decay_episodes)
        else:
            self.epsilon = self.final_epsilon

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state):
        state_idx = self.env.state_to_index(state)
        next_state_idx = self.env.state_to_index(next_state)

        old_value = self.q_table[state_idx, action]
        next_max = np.max(self.q_table[next_state_idx])

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        self.q_table[state_idx, action] = new_value


N, M = 10, 10
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000

env = MazeEnvironment(N, M)
agent = QLearningAgent(env)
rewards_history = []

print("Starting training...")
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    
    agent.random_action_counter = 0
    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.choose_action(state, episode)
        next_state, reward, done = env.step(state, action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if done:
            break
    
    rewards_history.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f'Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}')

print("Training completed.")
print('-' * 30)

print("Starting evaluation...")
NUM_TEST_EPISODES = 100
total_steps_list = []
successes = 0

for _ in range(NUM_TEST_EPISODES):
    state = env.reset()
    done = False
    steps_this_episode = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        state_idx = env.state_to_index(state)
        action = np.argmax(agent.q_table[state_idx])

        next_state, reward, done = env.step(state, action)

        state = next_state
        steps_this_episode += 1

        if done:
            if reward == 100:
                successes += 1
                total_steps_list.append(steps_this_episode)
            break

# Calculate and report metrics
success_rate = (successes / NUM_TEST_EPISODES) * 100
avg_steps = np.mean(total_steps_list) if successes > 0 else float('inf')

print(f"--- Evaluation Results (100 episodes) ---")
print(f"Success Rate: {success_rate:.2f}%")
print(f"Average Steps to Goal (on success): {avg_steps:.2f}")
print("-" * 30)


plt.figure(figsize=(12, 6))
plt.plot(rewards_history, label='Total Reward per Episode')


rolling_avg = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
plt.plot(range(99, NUM_EPISODES), rolling_avg, label='100-Episode Rolling Average', color='red', linewidth=2)

plt.title('Milestone 1: Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("maze1.png")