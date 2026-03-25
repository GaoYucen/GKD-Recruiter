import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import sys

# 在文件最上方确保导入了环境
from gkd_env import GKDEnv

# 假设 GKDEnv 已经定义在上一级目录或环境变量中可用
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from gkd_env import GKDEnv 

class VanillaDQN(nn.Module):
    """
    基础 DQN 网络 (Baseline: DQNSelector)
    纯 MLP 架构，不包含任何 RGCN/IGAT 图结构信息
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
        return (torch.stack(states), torch.tensor(actions), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.stack(next_states), torch.tensor(dones, dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)

def train_dqn_selector(env, episodes=50, batch_size=64):
    print("🚀 [Baseline] 开始训练 DQNSelector (无图结构的纯 MLP 强化学习)...")
    
    # 状态维度：极简状态 (剩余预算比例 + 当前整体满意度) -> 维度为 2
    state_dim = 2 
    # 动作维度：所有的 (worker, task) 组合
    action_dim = env.num_workers * env.num_tasks
    
    # 声明网络
    policy_net = VanillaDQN(state_dim, action_dim)
    target_net = VanillaDQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()
    
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.95
    gamma = 0.99
    
    for ep in range(episodes):
        # Env 返回的状态我们需要构造成 state_dim = 2 的 Tensor
        _ = env.reset()
        state = torch.tensor([1.0, 0.0], dtype=torch.float32) 
        
        ep_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy 探索
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
                    
            # 步进环境
            _, reward, done, final_ets = env.step(action)
            ep_reward += reward
            
            # 构造 next_state
            next_state = torch.tensor([1.0 - (env.current_step / env.budget_K), final_ets], dtype=torch.float32)
            
            # 存入 Buffer
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # 经验回放更新网络
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
                
        # 更新 Target Network
        if ep % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"   Episode {ep+1:02d} | ETS: {final_ets:.4f} | Epsilon: {epsilon:.2f} | Buffer: {len(buffer)}")

if __name__ == "__main__":
    # 初始化环境
    env = GKDEnv(env_dir='data/env_params') # 确保路径能找到您的 data 文件夹
    # 开始训练
    train_dqn_selector(env, episodes=20, batch_size=32)