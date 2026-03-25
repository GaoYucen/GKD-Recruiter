import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import torch
import torch.optim as optim
import random
import os

# 假设 VanillaDQN 和 ReplayBuffer 已经可以从上一个文件导入，这里为了完整性直接简写逻辑
from dqn_selector import VanillaDQN, ReplayBuffer
from models.gkd_env import GKDEnv

class IndependentAgent:
    """
    Independent Local Agent (representing a sub-node in MAIM).
    Each agent is responsible for task recruitment within its assigned jurisdiction.
    """
    def __init__(self, agent_id, state_dim, num_workers, task_indices):
        self.agent_id = agent_id
        self.task_indices = task_indices
        self.action_dim = num_workers * len(task_indices) # Can only assign workers to its own tasks
        
        self.policy_net = VanillaDQN(state_dim, self.action_dim).to(device)
        self.target_net = VanillaDQN(state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(capacity=5000)

def train_maim_lite(env, num_agents=5, episodes=50, batch_size=32):
    print(f"🚀 [Baseline] Training MAIM (Multi-Agent Reinforcement Learning, Agents={num_agents})...")
    
    state_dim = 2
    tasks_per_agent = env.num_tasks // num_agents
    
    # Instantiate multiple independent agents
    agents = []
    for i in range(num_agents):
        task_subset = list(range(i * tasks_per_agent, (i + 1) * tasks_per_agent))
        agents.append(IndependentAgent(i, state_dim, env.num_workers, task_subset))
        
    epsilon = 1.0
    gamma = 0.99
    loss_fn = torch.nn.MSELoss()
    
    for ep in range(episodes):
        _ = env.reset()
        state = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
        done = False
        
        # Track cumulative reward for each agent (for logging)
        ep_rewards = [0] * num_agents
        
        while not done:
            # Round-Robin execution: Simulating decentralized decision-making
            for agent in agents:
                if done: break
                
                # Agent decision making
                if random.random() < epsilon:
                    local_action = random.randint(0, agent.action_dim - 1)
                else:
                    with torch.no_grad():
                        q_vals = agent.policy_net(state)
                        local_action = torch.argmax(q_vals).item()
                
                # Decode local action to global action index
                w_idx = local_action // len(agent.task_indices)
                local_t_idx = local_action % len(agent.task_indices)
                global_task_id = agent.task_indices[local_t_idx]
                global_action = w_idx * env.num_tasks + global_task_id
                
                # Step environment
                _, reward, done, final_ets = env.step(global_action)
                ep_rewards[agent.agent_id] += reward
                next_state = torch.tensor([1.0 - (env.current_step / env.budget_K), final_ets], dtype=torch.float32, device=device)
                
                # Experience storage and update
                agent.buffer.push(state, local_action, reward, next_state, done)
                
                if len(agent.buffer) > batch_size:
                    b_s, b_a, b_r, b_ns, b_d = agent.buffer.sample(batch_size)
                    q_v = agent.policy_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        max_nq = agent.target_net(b_ns).max(1)[0]
                        target = b_r + gamma * max_nq * (1 - b_d)
                    loss = loss_fn(q_v, target)
                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()
                
                state = next_state
        
        # Target network update
        if ep % 5 == 0:
            for agent in agents:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                
        epsilon = max(0.05, epsilon * 0.95)
        print(f"   Episode {ep+1:02d} | 最终 ETS: {final_ets:.4f} | Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    # 初始化环境
    env = GKDEnv(env_dir='data/env_params') 
    # 开始训练
    train_maim_lite(env, num_agents=5, episodes=20, batch_size=32)