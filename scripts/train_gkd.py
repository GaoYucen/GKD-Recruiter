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

class ReplayBuffer:
    """Standard experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state_mask, action, reward, next_state_mask, done):
        self.buffer.append((state_mask, action, reward, next_state_mask, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_masks, actions, rewards, next_state_masks, dones = zip(*batch)
        return (
            torch.stack(state_masks),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_state_masks),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_rainbow_dqn():
    print("🌈 Starting Stage 2: Seed Selection via Rainbow DQN...")
    
    # ==========================================
    # 0. Enable hardware acceleration (Mac MPS and Nvidia CUDA support)
    # ==========================================
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🖥️  Enabling Mac Apple Silicon (MPS) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🖥️  Enabling CUDA acceleration!")
    else:
        device = torch.device("cpu")
        print("🖥️  Running on CPU.")
    
    # ==========================================
    # 1. Load high-quality distilled features from Stage 1
    # ==========================================
    print("📥 Loading distilled features...")
    try:
        # Add weights_only=True to fix PyTorch warning
        worker_embeds = torch.load('data/pretrain/distilled_worker_embeds.pt', weights_only=True).to(device)
        task_embeds = torch.load('data/pretrain/distilled_task_embeds.pt', weights_only=True).to(device)
    except FileNotFoundError:
        print("❌ Feature files not found. Please run train_representation.py first!")
        return
        
    num_w, hidden_dim = worker_embeds.shape
    num_t, _ = task_embeds.shape
    action_space_size = num_w * num_t 
    
    # ==========================================
    # 2. Section 5.6: Action Space Top-m Pruning
    # ==========================================
    m = 5
    print(f"✂️ Applying Top-{m} action space pruning (reducing complexity)...")
    with torch.no_grad():
        # Find the 5 most matching tasks for each worker via feature similarity
        affinity_matrix = torch.matmul(worker_embeds, task_embeds.t()) # [num_w, num_t]
        _, top_m_indices = torch.topk(affinity_matrix, m, dim=1) # [num_w, m]
        
    valid_actions = []
    for w in range(num_w):
        for t_idx in range(m):
            t = top_m_indices[w, t_idx].item()
            valid_actions.append(w * num_t + t) # Flattened 1D Action ID
            
    valid_actions_set = set(valid_actions)
    valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.long, device=device)
    print(f"📉 Action space dropped from {action_space_size} to {len(valid_actions)}! Training speed will improve significantly.")

    # ==========================================
    # 3. Initialize Rainbow DQN
    # ==========================================
    q_net = DuelingQNetwork(hidden_dim).to(device)
    target_net = DuelingQNetwork(hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    memory = ReplayBuffer(capacity=10000)
    
    epochs = 200
    budget = 100            # Corresponds to budget K in the paper
    batch_size = 64
    gamma = 0.99            
    epsilon_start = 1.0     
    epsilon_end = 0.05      
    epsilon_decay = 0.995   
    target_update_freq = 5  
    
    epsilon = epsilon_start
    print("🚀 Starting RL training loop...")
    
    for episode in range(epochs):
        chosen_actions = set()
        state_mask = torch.zeros(action_space_size, device=device)
        episode_reward = 0.0
        
        # --- Data Collection (Explore-Exploit) ---
        for step in range(budget):
            if random.random() < epsilon:
                # Explore randomly within legal Top-m actions
                available_actions = list(valid_actions_set - chosen_actions)
                action = random.choice(available_actions) if available_actions else random.choice(valid_actions)
            else:
                with torch.no_grad():
                    w_in = worker_embeds.unsqueeze(0)
                    t_in = task_embeds.unsqueeze(0)
                    q_values = q_net(w_in, t_in).squeeze(0) # [action_space_size]
                    
                    # Masking logic: filter non-Top-m actions and already selected actions
                    mask = torch.ones_like(q_values, dtype=torch.bool)
                    mask[valid_actions_tensor] = False # Set legal actions to False (unmasked)
                    
                    if chosen_actions:
                        chosen_tensor = torch.tensor(list(chosen_actions), dtype=torch.long, device=device)
                        mask[chosen_tensor] = True # Set selected actions to True (masked)
                        
                    q_values[mask] = -float('inf')
                    action = q_values.argmax().item()
            
            # Simulated environment Reward feedback (simple decay to simulate saturation trap)
            reward = random.uniform(0.1, 1.0) * (1.0 - step / budget) 
            done = (step == budget - 1)
            
            next_state_mask = state_mask.clone()
            next_state_mask[action] = 1.0
            
            memory.push(state_mask, action, reward, next_state_mask, done)
            
            state_mask = next_state_mask
            chosen_actions.add(action)
            episode_reward += reward

        # --- Batch training at the end of each episode for faster performance ---
        if len(memory) >= batch_size:
            # Perform 5 updates
            for _ in range(5):
                b_states, b_actions, b_rewards, b_next_states, b_dones = memory.sample(batch_size)
                b_actions = b_actions.to(device).unsqueeze(1)
                b_rewards = b_rewards.to(device)
                b_dones = b_dones.to(device)
                
                b_w_in = worker_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                b_t_in = task_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                
                current_q_values = q_net(b_w_in, b_t_in)
                current_q = current_q_values.gather(1, b_actions).squeeze(1)
                
                with torch.no_grad():
                    next_q_values = q_net(b_w_in, b_t_in)
                    next_q_values = next_q_values - (b_next_states.to(device) * 1e9)
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
            print(f"Episode: {episode + 1}/{epochs} | Total Reward (ETS): {episode_reward:.4f} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train_rainbow_dqn()
