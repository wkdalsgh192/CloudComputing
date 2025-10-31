import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class MazeEnvironment:
    """
    Modified to include a state_to_one_hot helper function
    for the neural network.
    """
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
        
    def state_to_one_hot(self, state):
        """
        Converts a 2D (row, col) state into a 1D one-hot tensor.
        """
        tensor = torch.zeros(1, self.state_space_size)
        index = self.state_to_index(state)
        tensor[0][index] = 1
        return tensor


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, env):
        self.env = env
        
        self.model = DQN(env.state_space_size, env.action_space_size)
        self.target_model = DQN(env.state_space_size, env.action_space_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.HuberLoss()
        
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        
        self.gamma = 0.9 
        
        self.initial_epsilon = 0.9
        self.final_epsilon = 0.1
        self.epsilon_decay_episodes = 500
        self.epsilon = self.initial_epsilon

    def choose_action(self, state, episode):
        if episode < self.epsilon_decay_episodes:
            self.epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (episode / self.epsilon_decay_episodes)
        else:
            self.epsilon = self.final_epsilon
            
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.action_space_size - 1)
        else:
            with torch.no_grad():
                state_tensor = self.env.state_to_one_hot(state)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()
                
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        state_tensors = torch.cat([self.env.state_to_one_hot(s) for s in states])
        next_state_tensors = torch.cat([self.env.state_to_one_hot(s) for s in next_states])
        action_tensors = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
        reward_tensors = torch.tensor(rewards, dtype=torch.float).unsqueeze(-1)
        done_tensors = torch.tensor(dones, dtype=torch.float).unsqueeze(-1) 
        all_current_q_values = self.model(state_tensors)
        current_q_values = torch.gather(all_current_q_values, 1, action_tensors)

        with torch.no_grad():
            all_next_q_values = self.target_model(next_state_tensors)
            next_max_q = all_next_q_values.max(1)[0].unsqueeze(-1)

        target_q_values = reward_tensors + (1 - done_tensors) * self.gamma * next_max_q

        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad() 
        loss.backward()            
        self.optimizer.step()      

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


N, M = 10, 10
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
TARGET_UPDATE_FREQUENCY = 10 

env = MazeEnvironment(N, M)
agent = DQNAgent(env)
rewards_history = []

print("Starting DQN training...")
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    
    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.choose_action(state, episode)
        next_state, reward, done = env.step(state, action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.update()
        
        if done:
            break
            
    if episode % TARGET_UPDATE_FREQUENCY == 0:
        agent.update_target_network()
            
    rewards_history.append(total_reward)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{NUM_EPISODES} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")

print("Training finished.")
print("-" * 30)
print("Starting evaluation...")
NUM_TEST_EPISODES = 100
total_steps_list = []
successes = 0

for _ in range(NUM_TEST_EPISODES):
    state = env.reset()
    done = False
    steps_this_episode = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        with torch.no_grad():
            state_tensor = env.state_to_one_hot(state)
            q_values = agent.model(state_tensor)
            action = torch.argmax(q_values).item()
        
        next_state, reward, done = env.step(state, action)
        
        state = next_state
        steps_this_episode += 1
        
        if done:
            if reward == 100: 
                successes += 1
                total_steps_list.append(steps_this_episode)
            break

success_rate = (successes / NUM_TEST_EPISODES) * 100
avg_steps = np.mean(total_steps_list) if successes > 0 else float('inf')

print(f"--- Evaluation Results (100 episodes) ---")
print(f"Success Rate: {success_rate:.2f}%")
print(f"Average Steps to Goal (on success): {avg_steps:.2f}")
print("-" * 30)

plt.figure(figsize=(12, 6))
plt.plot(rewards_history, label='Total Reward per Episode', alpha=0.3)

rolling_avg = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
plt.plot(range(99, NUM_EPISODES), rolling_avg, label='100-Episode Rolling Avg', color='red', linewidth=2)

plt.title('Milestone 3: DQN Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()