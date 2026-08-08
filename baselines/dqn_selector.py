import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

# 在文件最上方确保导入了环境
from models.gkd_env import GKDEnv

# 假设 GKDEnv 已经定义在上一级目录或环境变量中可用
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from gkd_env import GKDEnv 

class VanillaDQN(nn.Module):
    """
    Basic DQN Network (Baseline: DQNSelector)
    Pure MLP architecture, excluding any RGCN/IGAT graph structural information.
    """
    def __init__(self, state_dim, action_dim):
        super(VanillaDQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states).to(device), torch.tensor(actions, device=device), 
                torch.tensor(rewards, dtype=torch.float32, device=device), 
                torch.stack(next_states).to(device), torch.tensor(dones, dtype=torch.float32, device=device))
    
    def __len__(self):
        return len(self.buffer)

def train_dqn_selector(env, episodes=50, batch_size=64):
    print("🚀 [Baseline] Training DQNSelector (Pure MLP RL without graph structures)...")
    
    # State space: Minimalist state (budget ratio + current satisfaction) -> dim = 2
    state_dim = 2 
    # Action space: All (worker, task) combinations
    action_dim = env.num_workers * env.num_tasks
    
    # Initialize networks
    policy_net = VanillaDQN(state_dim, action_dim).to(device)
    target_net = VanillaDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()
    
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.95
    gamma = 0.99
    
    for ep in range(episodes):
        # Reset environment and construct initial state tensor
        _ = env.reset()
        state = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device) 
        
        ep_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy exploration
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
                    
            # Step environment
            _, reward, done, final_ets = env.step(action)
            ep_reward += reward
            
            # Construct next_state
            next_state = torch.tensor([1.0 - (env.current_step / env.budget_K), final_ets], dtype=torch.float32)
            
            # Save to Buffer
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # Experience Replay for network update
            if len(buffer) > batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size)
                
                q_values = policy_net(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(b_next_states).max(1)[0]
                    target_q = b_rewards + gamma * max_next_q * (1 - b_dones)
                    
                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Update Target Network
        if ep % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"   Episode {ep+1:02d} | ETS: {final_ets:.4f} | Epsilon: {epsilon:.2f} | Buffer: {len(buffer)}")

if __name__ == "__main__":
    # Initialize environment
    env = GKDEnv(env_dir='data/env_params') # Ensure the path reaches your data folder
    # Start training
    train_dqn_selector(env, episodes=20, batch_size=32)