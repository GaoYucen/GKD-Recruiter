import torch
import numpy as np
import os

class GKDDataPreprocessor:
    """
    Data Preprocessing Layer:
    Converts raw Source Data into Environment Parameters and Model Inputs.
    """
    def __init__(self, base_dir="data", embed_dim=64):
        self.base_dir = base_dir
        self.embed_dim = embed_dim
        self.dir_source = os.path.join(base_dir, 'source_data')
        self.dir_env = os.path.join(base_dir, 'env_params')
        self.dir_model = os.path.join(base_dir, 'model_inputs')

    def preprocess(self):
        print("🔍 Starting data preprocessing: Source Data -> Env Params & Model Inputs")
        
        # Create output directories
        for d in [self.dir_env, self.dir_model]:
            os.makedirs(d, exist_ok=True)

        # 1. Load Source Data
        print("   - Loading source data...")
        raw_edge_index = torch.tensor(np.loadtxt(os.path.join(self.dir_source, 'raw_edge_index.txt')), dtype=torch.long).t().contiguous()
        worker_locations = torch.tensor(np.loadtxt(os.path.join(self.dir_source, 'worker_locations.txt')), dtype=torch.float)
        task_locations = torch.tensor(np.loadtxt(os.path.join(self.dir_source, 'task_locations.txt')), dtype=torch.float)
        raw_visit_freq = torch.tensor(np.loadtxt(os.path.join(self.dir_source, 'raw_visit_freq.txt')), dtype=torch.float)

        num_nodes = int(raw_edge_index.max() + 1)
        num_workers = worker_locations.size(0)
        num_tasks = task_locations.size(0)

        # Map workers to node indices (Consistent with synthetic logic or pre-defined mapping)
        # Assuming the first num_workers nodes in the source data are candidates or identified in the dataset
        torch.manual_seed(42) 
        worker_indices = torch.randperm(num_nodes)[:num_workers]

        # 2. Generate Environmental Parameters (Env Params)
        print("   - Generating environment parameters...")
        dist_matrix = torch.cdist(worker_locations, task_locations)
        
        # Core formulas from the paper
        w_ij = torch.rand(raw_edge_index.size(1), dtype=torch.float) * 0.09 + 0.01
        q_matrix = 1.0 - (dist_matrix / (dist_matrix.max() + 1e-9))
        task_rewards = torch.rand(num_tasks) * 10 + 5 
        
        raw_a_matrix = task_rewards.unsqueeze(0) * (raw_visit_freq / (raw_visit_freq.max() + 1e-9))
        a_matrix = raw_a_matrix / (raw_a_matrix.max() + 1e-9)
        task_demands = torch.randint(10, 50, (num_tasks,), dtype=torch.float)

        # Save Env Params (Ground Truth for evaluator and baselines)
        np.savetxt(os.path.join(self.dir_env, 'edge_index.txt'), raw_edge_index.numpy().T, fmt='%d')
        np.savetxt(os.path.join(self.dir_env, 'worker_indices.txt'), worker_indices.numpy(), fmt='%d')
        np.savetxt(os.path.join(self.dir_env, 'w_ij.txt'), w_ij.numpy(), fmt='%.4f')
        np.savetxt(os.path.join(self.dir_env, 'q_matrix.txt'), q_matrix.numpy(), fmt='%.4f')
        np.savetxt(os.path.join(self.dir_env, 'a_matrix.txt'), a_matrix.numpy(), fmt='%.4f')
        np.savetxt(os.path.join(self.dir_env, 'task_demands.txt'), task_demands.numpy(), fmt='%d')
        np.savetxt(os.path.join(self.dir_env, 'task_rewards.txt'), task_rewards.numpy(), fmt='%.2f')

        # 3. Generate Model Prep Features (Model Inputs)
        print("   - Generating model input features...")
        node_features = torch.randn(num_nodes, self.embed_dim) * 0.1
        worker_features = torch.zeros(num_workers, self.embed_dim)
        task_features = torch.zeros(num_tasks, self.embed_dim)
        
        # Dim 0, 1: Geographic coordinates
        worker_features[:, 0:2] = worker_locations / 10.0
        task_features[:, 0:2] = task_locations / 10.0
        
        # Dim 2: Summary features
        worker_features[:, 2] = raw_visit_freq.sum(dim=1) / (raw_visit_freq.sum(dim=1).max() + 1e-9)
        task_features[:, 2] = task_demands / 50.0
        
        # Padding remaining dimensions
        worker_features[:, 3:] = torch.randn(num_workers, self.embed_dim - 3) * 0.1
        task_features[:, 3:] = torch.randn(num_tasks, self.embed_dim - 3) * 0.1
        
        node_features[worker_indices] = worker_features
        
        # RGCN Heterogeneous Edges (Top-m = 5)
        m = 5
        _, top_m_indices = torch.topk(q_matrix, k=m, dim=1)
        wt_edge_src = torch.arange(num_workers).view(-1, 1).repeat(1, m).view(-1)
        wt_edge_dst = top_m_indices.view(-1)
        
        wt_edge_index = torch.stack([wt_edge_src, wt_edge_dst], dim=0)
        tw_edge_index = torch.stack([wt_edge_dst, wt_edge_src], dim=0)
        
        hetero_edge_index = torch.cat([wt_edge_index, tw_edge_index], dim=1)
        hetero_edge_type = torch.cat([
            torch.zeros(wt_edge_index.size(1), dtype=torch.long),
            torch.ones(tw_edge_index.size(1), dtype=torch.long)
        ])

        # Similarity Matrices (Cosine Similarity)
        worker_sim_adj = torch.nn.functional.cosine_similarity(worker_features.unsqueeze(1), worker_features.unsqueeze(0), dim=2)
        task_sim_adj = torch.nn.functional.cosine_similarity(task_features.unsqueeze(1), task_features.unsqueeze(0), dim=2)

        # Save Model Inputs (Optimized for IGAT/RGCN)
        np.savetxt(os.path.join(self.dir_model, 'node_features.txt'), node_features.numpy(), fmt='%.4f')
        np.savetxt(os.path.join(self.dir_model, 'worker_features.txt'), worker_features.numpy(), fmt='%.4f')
        np.savetxt(os.path.join(self.dir_model, 'task_features.txt'), task_features.numpy(), fmt='%.4f')
        np.savetxt(os.path.join(self.dir_model, 'hetero_edge_index.txt'), hetero_edge_index.numpy().T, fmt='%d')
        np.savetxt(os.path.join(self.dir_model, 'hetero_edge_type.txt'), hetero_edge_type.numpy(), fmt='%d')
        np.savetxt(os.path.join(self.dir_model, 'worker_sim_adj.txt'), worker_sim_adj.numpy(), fmt='%.4f')
        np.savetxt(os.path.join(self.dir_model, 'task_sim_adj.txt'), task_sim_adj.numpy(), fmt='%.4f')
        
        print("✅ Preprocessing complete! Env params and model inputs are saved.")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    preprocessor = GKDDataPreprocessor()
    preprocessor.preprocess()
