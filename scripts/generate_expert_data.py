import sys
from pathlib import Path
# 自动将项目根目录加入搜索路径
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import os
import torch
import numpy as np
from tqdm import tqdm 
import networkx as nx

# Import environment and feature wrappers
from models.gkd_env import GKDEnv
from scripts.train_gkd import GKDGraphWrapper

def generate_comgreedy_expert_data(num_episodes=100, save_path='data/expert_data.pt'):
    print(f"🎓 Starting ComGreedy expert data collection (Total episodes: {num_episodes})...")
    
    # Initialize environment
    env = GKDEnv(env_dir='data/env_params')
    # Force Wrapper device to CPU to avoid memory issues on MPS during large-scale data generation
    from scripts import train_gkd
    train_gkd.device = torch.device("cpu") 
    wrapper = GKDGraphWrapper(env)
    # Manually reset wrapper internal tensors to CPU
    wrapper.ww_adj = wrapper.ww_adj.to(train_gkd.device)
    wrapper.wt_adj_full = wrapper.wt_adj_full.to(train_gkd.device)
    wrapper.base_node_feat = wrapper.base_node_feat.to(train_gkd.device)
    wrapper.base_t_feat = wrapper.base_t_feat.to(train_gkd.device)
    
    # 1. Precompute expert heuristic scores
    # ComGreedy core: worker social influence (sum of out-edge weights) * task affinity
    print("   => Precomputing expert heuristic scores...")
    worker_influence = np.zeros(env.num_workers)
    for i, w_id in enumerate(env.worker_indices):
        # Compute expected propagation range (sum of out-edge weights)
        worker_influence[i] = sum([data.get('weight', 0.1) for _, _, data in env.G.out_edges(w_id, data=True)])
        
    # [300, 100] static score matrix: influence * quality * preference
    static_expert_score = worker_influence[:, None] * (env.q_matrix * env.a_matrix)
    
    # Lists to store expert trajectories
    expert_w_states = []
    expert_t_states = []
    expert_actions = []
    
    best_ets_list = []

    # 2. Expert rollout
    for ep in tqdm(range(num_episodes), desc="Expert rollout"):
        env.reset()
        done = False
        
        # Add small random noise (0.05) to expert data for diversity
        noise = np.random.uniform(0.95, 1.05, size=static_expert_score.shape)
        current_ep_score = static_expert_score * noise
        
        # Track task allocation count to avoid saturation
        task_alloc_count = np.zeros(env.num_tasks)
        
        while not done:
            # Get current dynamic graph features
            w_f, t_f = wrapper.get_dynamic_graph_state()
            
            # --- ComGreedy Expert Decision Engine ---
            # Dynamic discount based on current task progress: avoid over-saturated tasks
            dynamic_discount = np.ones(env.num_tasks)
            for t in range(env.num_tasks):
                # If a task already has 3 workers assigned, significantly reduce reward
                if task_alloc_count[t] >= 3: 
                    dynamic_discount[t] = 0.01 
                    
            # Compute step scores [300, 100]
            step_score = current_ep_score * dynamic_discount[None, :]
            
            # Select the pair (worker, task) with the highest score
            best_flat_idx = np.argmax(step_score)
            best_w_local_idx = best_flat_idx // env.num_tasks
            best_t_idx = best_flat_idx % env.num_tasks
            
            # Record trajectory (save to CPU memory)
            expert_w_states.append(w_f.clone())
            expert_t_states.append(t_f.clone())
            expert_actions.append(best_flat_idx)
            
            # Execute action
            _, _, done, final_ets = env.step(best_flat_idx)
            
            # Update local trackers
            task_alloc_count[best_t_idx] += 1
            # Reset score for the selected worker (no duplicate recruitment)
            current_ep_score[best_w_local_idx, :] = -1.0 
            
        best_ets_list.append(final_ets)

    print(f"\n📊 Expert data collection finished! Mean ETS: {np.mean(best_ets_list):.4f}")
    
    # 3. Pack and save to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    expert_dataset = {
        'w_states': torch.stack(expert_w_states),   
        't_states': torch.stack(expert_t_states),   
        'actions': torch.tensor(expert_actions, dtype=torch.long) 
    }
    torch.save(expert_dataset, save_path)
    print(f"💾 Expert knowledge saved to: {save_path} (Total steps: {len(expert_actions)})")
    
    # 3. 打包并保存到磁盘
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    expert_dataset = {
        'w_states': torch.stack(expert_w_states),   # [5000, 3000, 17]
        't_states': torch.stack(expert_t_states),   # [5000, 100, 17]
        'actions': torch.tensor(expert_actions, dtype=torch.long) # [5000]
    }
    torch.save(expert_dataset, save_path)
    print(f"💾 专家知识已压缩并保存至: {save_path} (数据量: {len(expert_actions)} 步)")

if __name__ == "__main__":
    generate_comgreedy_expert_data(num_episodes=100)