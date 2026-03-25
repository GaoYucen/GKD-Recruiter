# gkd_env.py
import numpy as np
import networkx as nx
import os
from .evaluate import GKDEvaluator

class GKDEnv:
    def __init__(self, env_dir='data/env_params', budget_K=50): # Note: adjust the path according to your actual directory structure
        self.budget_K = budget_K
        self.q_matrix = np.loadtxt(os.path.join(env_dir, 'q_matrix.txt'))
        self.a_matrix = np.loadtxt(os.path.join(env_dir, 'a_matrix.txt'))
        self.task_demands = np.loadtxt(os.path.join(env_dir, 'task_demands.txt'))
        self.worker_indices = np.loadtxt(os.path.join(env_dir, 'worker_indices.txt'), dtype=int)
        
        edge_index = np.loadtxt(os.path.join(env_dir, 'edge_index.txt'), dtype=int)
        w_ij = np.loadtxt(os.path.join(env_dir, 'w_ij.txt'))
        self.G = nx.DiGraph()
        self.G.add_edges_from([(edge_index[i][0], edge_index[i][1], {'weight': w_ij[i]}) for i in range(len(w_ij))])
        
        # Accelerate training using a small number of simulations (5 times)
        self.evaluator = GKDEvaluator(self.G, self.q_matrix, self.a_matrix, self.task_demands, self.worker_indices, num_simulations=5)
        
        self.num_workers = len(self.worker_indices)
        self.num_tasks = len(self.task_demands)

    def reset(self):
        self.current_step = 0
        self.selected_seeds = []
        self.current_ets = 0.0
        import torch
        return torch.tensor([1.0, 0.0], dtype=torch.float32)

    def step(self, action_idx):
        w_local_idx = action_idx // self.num_tasks
        task_id = action_idx % self.num_tasks
        worker_id = self.worker_indices[w_local_idx]
        
        self.selected_seeds.append((worker_id, task_id))
        self.current_step += 1
        
        new_results = self.evaluator.evaluate(self.selected_seeds)
        new_ets = new_results['Effective_Task_Satisfaction']
        reward = new_ets - self.current_ets
        self.current_ets = new_ets
        
        done = (self.current_step >= self.budget_K)
        import torch
        next_state = torch.tensor([1.0 - (self.current_step / self.budget_K), new_ets], dtype=torch.float32)
        
        return next_state, reward, done, new_ets