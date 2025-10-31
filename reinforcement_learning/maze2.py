import numpy as np
import random
import matplotlib.pyplot as plt
import copy

class MazeEnvironment:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.state_space_size = N * M
        self.action_space_size = 4
        self.start = (0, 0)
        self.goal = (N - 1, M - 1)
        self.grid = np.zeros((N, M))
        self.grid[self.goal] = 2
        num_walls = int(self.state_space_size * 0.1)
        walls_placed = 0
        while walls_placed < num_walls:
            r, c = random.randint(0, N-1), random.randint(0, M-1)
            if (r, c) != self.start and (r, c) != self.goal and self.grid[r, c] == 0:
                self.grid[r, c] = 1
                walls_placed += 1
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self):
        return self.start

    def step(self, state, action):
        row, col = state
        d_row, d_col = self.action_map[action]
        new_row, new_col = row + d_row, col + d_col
        done = False
        if new_row < 0 or new_row >= self.N or \
           new_col < 0 or new_col >= self.M or \
           self.grid[new_row, new_col] == 1:
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


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.random.uniform(-0.1, 0.1, (env.state_space_size, env.action_space_size))
        
        
        self.alpha = 0.1 
        self.gamma = 0.9 
        
        
        self.initial_epsilon = 0.5 
        self.final_epsilon = 0.1
        self.epsilon_decay_episodes = 500  
        self.epsilon = self.initial_epsilon
        
        self.random_action_counter = 0

    def choose_action(self, state, episode, num_decay_episodes):
        state_idx = self.env.state_to_index(state)


        if episode > 0 and episode % 5 == 0 and self.random_action_counter < 3:
            self.random_action_counter += 1
            return random.randint(0, self.env.action_space_size - 1)
        
        if episode % 5 != 0:
            self.random_action_counter = 0
            
            
        if episode < num_decay_episodes:
            self.epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (episode / num_decay_episodes)
        else:
            self.epsilon = self.final_epsilon
            
            
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state_idx])

    def update(self, state, action, reward, next_state, is_historical_state=False):
        state_idx = self.env.state_to_index(state)
        next_state_idx = self.env.state_to_index(next_state)
        
        current_alpha = 0.15 if is_historical_state else self.alpha
        
        old_value = self.q_table[state_idx, action]
        next_max = np.max(self.q_table[next_state_idx])
        
        new_value = old_value + current_alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_idx, action] = new_value


def evaluate_agent(agent, env, num_episodes, max_steps):
    total_steps_list = []
    successes = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps_this_episode = 0
        for _ in range(max_steps):
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
    success_rate = (successes / num_episodes) * 100
    avg_steps = np.mean(total_steps_list) if successes > 0 else float('inf')
    return success_rate, avg_steps


N, M = 10, 10
M1_DATA_EPISODES = 500 
M2_TRAIN_EPISODES = 500 
MAX_STEPS_PER_EPISODE = 1000 
GOOD_EPISODE_THRESHOLD = 200 
NUM_TEST_EPISODES = 100 

env = MazeEnvironment(N, M)
historical_data = []
m1_rewards_history = []

print(f"Starting Phase 1: Generating historical data from M1 agent ({M1_DATA_EPISODES} episodes)...")
m1_agent = QLearningAgent(env)

m1_agent.initial_epsilon = 0.9
m1_agent.epsilon_decay_episodes = 500
m1_agent.epsilon = m1_agent.initial_epsilon

for episode in range(M1_DATA_EPISODES):
    episode_trajectory = []
    state = env.reset()
    total_reward = 0
    reward = 0
    done = False
    steps_this_episode = 0
    
    m1_agent.random_action_counter = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        action = m1_agent.choose_action(state, episode, num_decay_episodes=M1_DATA_EPISODES)
        next_state, reward, done = env.step(state, action)
        
        m1_agent.update(state, action, reward, next_state) 
        
        episode_trajectory.append((state, action, reward, next_state))
        state = next_state
        total_reward += reward
        steps_this_episode += 1
        
        if done:
            break
            
    m1_rewards_history.append(total_reward)
    
    if done and reward == 100 and steps_this_episode < GOOD_EPISODE_THRESHOLD:
        historical_data.extend(episode_trajectory)

print(f"Phase 1 complete. Generated {len(historical_data)} historical data points.")
print("-" * 30)

print("Starting Phase 2: Training Milestone 2 Agent...")
m2_agent = QLearningAgent(env)
m2_rewards_history = []

historical_states = set([s for (s, a, r, s_next) in historical_data])
print(f"Found {len(historical_states)} unique states in historical data.")

print(f"Pre-loading Q-Table with {len(historical_data)} data points...")
for state, action, reward, next_state in historical_data:
    m2_agent.update(state, action, reward, next_state, is_historical_state=True)
print("Pre-loading complete.")

print(f"Starting self-play for {M2_TRAIN_EPISODES} episodes...")
for episode in range(M2_TRAIN_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    
    m2_agent.random_action_counter = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        action = m2_agent.choose_action(state, episode, num_decay_episodes=M2_TRAIN_EPISODES)
        next_state, reward, done = env.step(state, action)
        
        is_hist = (state in historical_states)
        m2_agent.update(state, action, reward, next_state, is_hist)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    m2_rewards_history.append(total_reward)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{M2_TRAIN_EPISODES} - Total Reward: {total_reward} - Epsilon: {m2_agent.epsilon:.3f}")

print("Phase 2 training finished.")
print("-" * 30)

print("Evaluating agents...")

m1_success, m1_steps = evaluate_agent(m1_agent, env, NUM_TEST_EPISODES, MAX_STEPS_PER_EPISODE)
print(f"--- Milestone 1 Agent Results (500 episodes) ---")
print(f"Success Rate: {m1_success:.2f}%")
print(f"Average Steps to Goal: {m1_steps:.2f}")

m2_success, m2_steps = evaluate_agent(m2_agent, env, NUM_TEST_EPISODES, MAX_STEPS_PER_EPISODE)
print(f"--- Milestone 2 Agent Results ---")
print(f"Success Rate: {m2_success:.2f}%")
print(f"Average Steps to Goal: {m2_steps:.2f}")
print("-" * 30)

plt.figure(figsize=(14, 7))
def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window = 50
m1_rolling_avg = rolling_average(m1_rewards_history, window)
m2_rolling_avg = rolling_average(m2_rewards_history, window)

plt.plot(m1_rewards_history, label='M1 Agent Reward', color='blue', alpha=0.2)
plt.plot(m2_rewards_history, label='M2 Agent Reward', color='green', alpha=0.2)

plt.plot(range(window - 1, M1_DATA_EPISODES), m1_rolling_avg, label=f'M1 Agent ({window}-ep Rolling Avg)', color='blue', linewidth=2)
plt.plot(range(window - 1, M2_TRAIN_EPISODES), m2_rolling_avg, label=f'M2 Agent ({window}-ep Rolling Avg)', color='green', linewidth=2)

plt.title('Milestone 1 vs. Milestone 2: Training Rewards Comparison')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("maze2.png")
