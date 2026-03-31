import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Ensure models module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gkd_recruiter import DuelingQNetwork
# 🌟 Fix import: use the actual class name GKDEnv in your environment
from models.gkd_env import GKDEnv 

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, action_mask, action, reward, next_action_mask, done):
        self.buffer.append((action_mask, action, reward, next_action_mask, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        action_masks, actions, rewards, next_action_masks, dones = zip(*batch)
        return (
            torch.stack(action_masks),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_action_masks),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_rainbow_dqn():
    print("🌈 Starting Stage 2: Seed Selection via Rainbow DQN...")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("📥 Loading real distilled features...")
    try:
        worker_embeds = torch.load('data/pretrain/distilled_worker_embeds.pt', weights_only=True).to(device)
        task_embeds = torch.load('data/pretrain/distilled_task_embeds.pt', weights_only=True).to(device)
    except FileNotFoundError:
        print("❌ Feature files not found. Please run train_representation.py first!")
        return
        
    num_w, hidden_dim = worker_embeds.shape
    num_t, _ = task_embeds.shape
    action_space_size = num_w * num_t 
    
    # ==========================================
    # Top-m Action Space Pruning (m=5)
    # ==========================================
    m = 5
    print(f"✂️ Applying Top-{m} action space pruning...")
    with torch.no_grad():
        affinity_matrix = torch.matmul(worker_embeds, task_embeds.t()) 
        _, top_m_indices = torch.topk(affinity_matrix, m, dim=1) 
        
    valid_actions = []
    for w in range(num_w):
        for t_idx in range(m):
            t = top_m_indices[w, t_idx].item()
            valid_actions.append(w * num_t + t) 
            
    valid_actions_set = set(valid_actions)
    valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.long, device=device)

    # Initialize Model
    q_net = DuelingQNetwork(hidden_dim).to(device)
    target_net = DuelingQNetwork(hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    memory = ReplayBuffer(capacity=10000)
    
    epochs = 200
    budget = 100            
    batch_size = 64
    gamma = 0.99            
    epsilon = 1.0     
    epsilon_end = 0.05      
    epsilon_decay = 0.995   
    target_update_freq = 5  
    
    # ==========================================
    # 🌟 Fix initialization: use the actual parameters for your environment
    # ==========================================
    env = GKDEnv(env_dir='data/env_params', budget_K=budget) 
    
    print("🚀 Starting real RL training loop...")
    for episode in range(epochs):
        chosen_actions = set()
        episode_reward = 0.0
        
        # Reset environment (returns [budget, ets] which we can temporarily ignore for DQN feature input, as features mainly come from graph embeddings)
        env.reset() 
        # Maintain action mask independently to avoid dimension collapse
        action_mask = torch.zeros(action_space_size, device=device)
        
        for step in range(budget):
            # 1. Action Selection
            if random.random() < epsilon:
                available_actions = list(valid_actions_set - chosen_actions)
                action = random.choice(available_actions) if available_actions else random.choice(valid_actions)
            else:
                with torch.no_grad():
                    w_in = worker_embeds.unsqueeze(0)
                    t_in = task_embeds.unsqueeze(0)
                    q_values = q_net(w_in, t_in).squeeze(0) 
                    
                    # Mask non-Top-m actions
                    mask = torch.ones_like(q_values, dtype=torch.bool)
                    mask[valid_actions_tensor] = False 
                    
                    # Mask already chosen actions
                    if chosen_actions:
                        chosen_tensor = torch.tensor(list(chosen_actions), dtype=torch.long, device=device)
                        mask[chosen_tensor] = True 
                        
                    q_values[mask] = -float('inf')
                    action = q_values.argmax().item()
            
            # 2. Interaction with environment (Environment Step)
            # Based on gkd_env.py, returned values: next_state, reward, done, new_ets
            _, reward, done, current_ets = env.step(action)
            reward = float(reward)
            
            # Update action mask
            next_action_mask = action_mask.clone()
            next_action_mask[action] = 1.0
            
            memory.push(action_mask, action, reward, next_action_mask, done)
            
            action_mask = next_action_mask
            chosen_actions.add(action)
            episode_reward += reward
            
            if done:
                break

        # 3. Experience Replay and Network Updates (Batch Training)
        if len(memory) >= batch_size:
            for _ in range(5):
                b_masks, b_actions, b_rewards, b_next_masks, b_dones = memory.sample(batch_size)
                b_actions = b_actions.to(device).unsqueeze(1)
                b_rewards = b_rewards.to(device)
                b_dones = b_dones.to(device)
                
                b_w_in = worker_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                b_t_in = task_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                
                current_q_values = q_net(b_w_in, b_t_in)
                current_q = current_q_values.gather(1, b_actions).squeeze(1)
                
                with torch.no_grad():
                    next_q_values = q_net(b_w_in, b_t_in)
                    # Penalize already chosen actions to prevent invalid Q-value influence
                    next_q_values = next_q_values - (b_next_masks.to(device) * 1e9)
                    next_actions = next_q_values.argmax(dim=1, keepdim=True)
                    
                    target_q_values = target_net(b_w_in, b_t_in)
                    max_next_q = target_q_values.gather(1, next_actions).squeeze(1)
                    target_q = b_rewards + (gamma * max_next_q * (1 - b_dones))
                
                loss = F.mse_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
            
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1}/{epochs} | Real ETS: {current_ets:.4f} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train_rainbow_dqn()