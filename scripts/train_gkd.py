import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import networkx as nx

import sys
from pathlib import Path
# 自动将项目根目录加入搜索路径，确保跨文件夹 import 能通
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# Import models and environment
from models.gkd_env import GKDEnv
from models.gkd_recruiter import GKDRecruiterModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚡️ Current Compute Backend: {device}")

class GraphReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, dyn_w_feat, dyn_t_feat, action, reward, next_dyn_w, next_dyn_t, done):
        self.buffer.append((dyn_w_feat, dyn_t_feat, action, reward, next_dyn_w, next_dyn_t, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        w_f, t_f, act, rew, nw_f, nt_f, don = zip(*batch)
        # Convert batch to tensor for acceleration
        return (torch.stack(w_f).to(device), torch.stack(t_f).to(device), 
                torch.tensor(act, device=device), torch.tensor(rew, dtype=torch.float32, device=device), 
                torch.stack(nw_f).to(device), torch.stack(nt_f).to(device), 
                torch.tensor(don, dtype=torch.float32, device=device))
    
    def __len__(self):
        return len(self.buffer)

class GKDGraphWrapper:
    def __init__(self, env):
        self.env = env
        self.num_nodes = env.G.number_of_nodes()
        self.num_tasks = env.num_tasks
        # Pre-convert and fix to device to avoid redundant overhead
        self.ww_adj = torch.tensor(nx.to_numpy_array(env.G), dtype=torch.float32, device=device)
        self.wt_adj_full = torch.zeros(self.num_nodes, self.num_tasks, device=device)
        self.wt_adj_full[env.worker_indices] = torch.tensor(env.q_matrix * env.a_matrix, dtype=torch.float32, device=device)
        self.base_node_feat = torch.randn(self.num_nodes, 16, device=device)
        self.base_t_feat = torch.randn(self.num_tasks, 16, device=device)

    def get_dynamic_graph_state(self):
        node_selected = torch.zeros(self.num_nodes, 1, device=device)
        for w_id, _ in self.env.selected_seeds:
            node_selected[w_id] = 1.0
        t_progress = torch.zeros(self.num_tasks, 1, device=device)
        for _, t_id in self.env.selected_seeds:
            t_progress[t_id] += 0.05
        return torch.cat([self.base_node_feat, node_selected], dim=1), torch.cat([self.base_t_feat, t_progress], dim=1)


from torch.utils.data import TensorDataset, DataLoader

def pretrain_with_distillation(model, wrapper, worker_idx_tensor, data_path='data/expert_data.pt', epochs=15, batch_size=4):
    print(f"\n🎓 Starting GKD Knowledge Distillation Pre-training (Loading: {data_path})...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expert data not found: {data_path}. Please run generate_expert_data.py first.")
        
    # 1. Load data
    expert_data = torch.load(data_path, weights_only=True)
    dataset = TensorDataset(expert_data['w_states'], expert_data['t_states'], expert_data['actions'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() # Convert Q-value prediction to multi-class classification problem
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over replay buffer
        for b_w, b_t, b_act in dataloader:
            b_w, b_t, b_act = b_w.to(device), b_t.to(device), b_act.to(device)
            
            optimizer.zero_grad()
            # Model outputs full Q-values [Batch, 30000]
            q_values = model(b_w, b_t, wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
            
            # Cross Entropy Loss (ensuring expert actions have significantly higher Q-values)
            loss = criterion(q_values, b_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Distillation Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
    print("✅ Pre-training complete! GKD-Recruiter now possesses baseline expert knowledge.")
    return model

# def train_gkd_agent(episodes=50, batch_size=16, update_every=5):
#     env = GKDEnv(env_dir='data/env_params')
#     wrapper = GKDGraphWrapper(env)
    
#     # 🌟 Fix: Define global graph indices for model slicing
#     worker_idx_tensor = torch.tensor(env.worker_indices, dtype=torch.long, device=device)
    
#     model = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
#     target_net = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
#     target_net.load_state_dict(model.state_dict())
    
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     buffer = GraphReplayBuffer()
    
#     epsilon, epsilon_min, decay = 1.0, 0.1, 0.95
#     gamma = 0.99
#     global_step = 0

# Modify function signature to receive pre-trained model and environment parameters
def train_gkd_agent(model, env, wrapper, worker_idx_tensor, episodes=50, batch_size=16, update_every=5, initial_epsilon=0.1):
    print("\n🚀 Starting RL Fine-tuning (Exploring to surpass expert performance)...")
    
    target_net = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
    target_net.load_state_dict(model.state_dict())
    
    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=5e-5) 
    buffer = GraphReplayBuffer()
    
    # Epsilon starts from a lower value (e.g., 0.1)
    epsilon, epsilon_min, decay = initial_epsilon, 0.01, 0.95
    gamma = 0.99
    global_step = 0

    for ep in range(episodes):
        env.reset()
        w_f, t_f = wrapper.get_dynamic_graph_state()
        done = False
        
        while not done:
            global_step += 1
            # 1. Exploration and Decision Making
            if random.random() < epsilon:
                action = random.randint(0, env.num_workers * env.num_tasks - 1)
            else:
                with torch.no_grad():
                    # Add Batch dimension for inference [1, 3000, 17]
                    q_vals = model(w_f.unsqueeze(0), t_f.unsqueeze(0), wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
                    action = torch.argmax(q_vals.view(-1)).item()
            
            _, reward, done, final_ets = env.step(action)
            nw_f, nt_f = wrapper.get_dynamic_graph_state()
            buffer.push(w_f, t_f, action, reward, nw_f, nt_f, done)
            w_f, t_f = nw_f, nt_f

            # 2. Vectorized Batch Update (Performance core optimization)
            if len(buffer) > batch_size and global_step % update_every == 0:
                b_w, b_t, b_act, b_rew, b_nw, b_nt, b_don = buffer.sample(batch_size)
                
                # Calculate Q-values for the entire Batch [Batch, 30000]
                q_all = model(b_w, b_t, wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
                current_q = q_all.gather(1, b_act.unsqueeze(1)).squeeze(1)
                
                # Calculate Target Q-values
                with torch.no_grad():
                    next_q_all = target_net(b_nw, b_nt, wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
                    max_next_q = next_q_all.max(dim=1)[0]
                    target_q = b_rew + gamma * max_next_q * (1 - b_don)
                
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update strategy
        if ep % 5 == 0:
            target_net.load_state_dict(model.state_dict())
            if torch.backends.mps.is_available():
                torch.mps.empty_cache() # Release memory
        
        epsilon = max(epsilon_min, epsilon * decay)
        print(f"   Episode {ep+1:02d} | ETS: {final_ets:.4f} | Epsilon: {epsilon:.2f} | Buffer: {len(buffer)}")

# if __name__ == "__main__":
#     train_gkd_agent()


if __name__ == "__main__":
    # 1. Initialize unified environment and wrapper
    env = GKDEnv(env_dir='data/env_params')
    wrapper = GKDGraphWrapper(env)
    worker_idx_tensor = torch.tensor(env.worker_indices, dtype=torch.long, device=device)
    
    # 2. Instantiate a fresh model
    gkd_model = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
    
    # 🌟 Phase 1: Graph Knowledge Distillation (Pre-training)
    gkd_model = pretrain_with_distillation(gkd_model, wrapper, worker_idx_tensor, epochs=15)
    
    # 🌟 阶段二：强化学习微调 (RL Fine-tuning)
    train_gkd_agent(gkd_model, env, wrapper, worker_idx_tensor, episodes=30, initial_epsilon=0.1)
    
    # 保存最终称霸的王者模型！
    torch.save(gkd_model.state_dict(), 'data/gkd_recruiter_final.pth')
    print("🎉 全部训练闭环完成！最佳模型已保存！")