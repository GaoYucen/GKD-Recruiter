import torch
import networkx as nx
import numpy as np
import os

class GKDDatasetSimulator:
    """
    Generate synthetic spatial crowdsourcing datasets, strictly following a 
    three-tier architecture: Source Data -> Env Params -> Model Inputs.
    """
    def __init__(self, num_nodes=3000, num_workers=300, num_tasks=100, embed_dim=64):
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        
    def generate_and_save(self, base_dir="data"):
        print(f"🚀 Generating synthetic data: |V|={self.num_nodes}, |U|={self.num_workers}, |T|={self.num_tasks}")
        
        # Create directory hierarchy
        dir_source = os.path.join(base_dir, 'source_data')
        dir_env = os.path.join(base_dir, 'env_params')
        dir_model = os.path.join(base_dir, 'model_inputs')
        for d in [dir_source, dir_env, dir_model]:
            os.makedirs(d, exist_ok=True)

        # ==========================================
        # 1. Source Data Generation (Raw physics-based data)
        # ==========================================
        # 1.1 Social network topology
        G = nx.barabasi_albert_graph(self.num_nodes, m=3).to_directed()
        raw_edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        worker_indices = torch.randperm(self.num_nodes)[:self.num_workers]

        # 1.2 Physical coordinates (Simulating urban multi-center hotspots)
        num_hotspots = 5
        hotspots = torch.rand(num_hotspots, 2) * 8 + 1 # Generate 5 business centers
        
        # Distribute workers and tasks around hotspots (Gaussian distribution)
        worker_centers = hotspots[torch.randint(0, num_hotspots, (self.num_workers,))]
        task_centers = hotspots[torch.randint(0, num_hotspots, (self.num_tasks,))]
        
        worker_locations = worker_centers + torch.randn(self.num_workers, 2) * 1.5
        task_locations = task_centers + torch.randn(self.num_tasks, 2) * 1.5
        # Clamp to 10x10 grid
        worker_locations = torch.clamp(worker_locations, 0, 10)
        task_locations = torch.clamp(task_locations, 0, 10)

        # 1.3 Calculate distance and derived visit frequencies based on geographic decay
        dist_matrix = torch.cdist(worker_locations, task_locations)
        # Higher visit probability for closer proximity (exponential decay)
        prob_visit = torch.exp(-dist_matrix / 2.0) 
        # Simulate check-in frequency using Poisson distribution
        raw_visit_freq = torch.poisson(prob_visit * 10)
        raw_visit_freq = torch.clamp(raw_visit_freq, 0, 10)
        
        # Save Source Data (Simulating raw crawled information)
        np.savetxt(f'{dir_source}/raw_edge_index.txt', raw_edge_index.numpy().T, fmt='%d')
        np.savetxt(f'{dir_source}/worker_locations.txt', worker_locations.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_source}/task_locations.txt', task_locations.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_source}/raw_visit_freq.txt', raw_visit_freq.numpy(), fmt='%d')

        # ==========================================
        # 2. Environmental Parameters (Env Params)
        # ==========================================
        w_ij = torch.rand(raw_edge_index.size(1), dtype=torch.float) * 0.09 + 0.01
        q_matrix = 1.0 - (dist_matrix / dist_matrix.max())
        task_rewards = torch.rand(self.num_tasks) * 10 + 5 
        
        raw_a_matrix = task_rewards.unsqueeze(0) * (raw_visit_freq / (raw_visit_freq.max() + 1e-9))
        a_matrix = raw_a_matrix / (raw_a_matrix.max() + 1e-9)
        task_demands = torch.randint(10, 50, (self.num_tasks,), dtype=torch.float)

        # Save Env Params (Ground truth for evaluators and all baselines)
        np.savetxt(f'{dir_env}/edge_index.txt', raw_edge_index.numpy().T, fmt='%d')
        np.savetxt(f'{dir_env}/worker_indices.txt', worker_indices.numpy(), fmt='%d')
        np.savetxt(f'{dir_env}/w_ij.txt', w_ij.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_env}/q_matrix.txt', q_matrix.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_env}/a_matrix.txt', a_matrix.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_env}/task_demands.txt', task_demands.numpy(), fmt='%d')
        np.savetxt(f'{dir_env}/task_rewards.txt', task_rewards.numpy(), fmt='%.2f')

        # ==========================================
        # 3. Model Prep Features (Model Inputs) - Injecting Physical Semantics
        # ==========================================
        # 3.1 Initial Node Feature Vectors (Injecting physical properties into the first few dimensions)
        node_features = torch.randn(self.num_nodes, self.embed_dim) * 0.1
        
        # Extract features for workers and tasks
        worker_features = torch.zeros(self.num_workers, self.embed_dim)
        task_features = torch.zeros(self.num_tasks, self.embed_dim)
        
        # Dim 0, 1: Geographic coordinates (allowing GNN to learn spatial proximity)
        worker_features[:, 0:2] = worker_locations / 10.0
        task_features[:, 0:2] = task_locations / 10.0
        
        # Dim 2: Individual activity / Task budget scale
        worker_features[:, 2] = raw_visit_freq.sum(dim=1) / raw_visit_freq.sum(dim=1).max()
        task_features[:, 2] = task_demands / 50.0
        
        # 剩余维度填充小微噪声作为特征空间的探索裕度
        worker_features[:, 3:] = torch.randn(self.num_workers, self.embed_dim - 3) * 0.1
        task_features[:, 3:] = torch.randn(self.num_tasks, self.embed_dim - 3) * 0.1
        
        # 将构造好的 worker_features 覆盖回全局 node_features 中对应的位置
        node_features[worker_indices] = worker_features
        
        # 3.2 RGCN 异构图边修建 (Top-m = 5)
        m = 5
        _, top_m_indices = torch.topk(q_matrix, k=m, dim=1)
        wt_edge_src = torch.arange(self.num_workers).view(-1, 1).repeat(1, m).view(-1)
        wt_edge_dst = top_m_indices.view(-1)
        
        wt_edge_index = torch.stack([wt_edge_src, wt_edge_dst], dim=0)
        tw_edge_index = torch.stack([wt_edge_dst, wt_edge_src], dim=0)
        
        hetero_edge_index = torch.cat([wt_edge_index, tw_edge_index], dim=1)
        hetero_edge_type = torch.cat([
            torch.zeros(wt_edge_index.size(1), dtype=torch.long),
            torch.ones(tw_edge_index.size(1), dtype=torch.long)
        ])

        # 3.3 Correlation Layer 相似度矩阵 (余弦相似度)
        worker_sim_adj = torch.nn.functional.cosine_similarity(worker_features.unsqueeze(1), worker_features.unsqueeze(0), dim=2)
        task_sim_adj = torch.nn.functional.cosine_similarity(task_features.unsqueeze(1), task_features.unsqueeze(0), dim=2)

        # 保存 Model Inputs (仅供神经网络加速计算使用)
        np.savetxt(f'{dir_model}/node_features.txt', node_features.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_model}/worker_features.txt', worker_features.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_model}/task_features.txt', task_features.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_model}/hetero_edge_index.txt', hetero_edge_index.numpy().T, fmt='%d')
        np.savetxt(f'{dir_model}/hetero_edge_type.txt', hetero_edge_type.numpy(), fmt='%d')
        np.savetxt(f'{dir_model}/worker_sim_adj.txt', worker_sim_adj.numpy(), fmt='%.4f')
        np.savetxt(f'{dir_model}/task_sim_adj.txt', task_sim_adj.numpy(), fmt='%.4f')
        
        print("✅ 数据生成完毕！已成功划分为 source_data, env_params, model_inputs 三层目录。")

if __name__ == "__main__":
    # 设置随机种子保证每次合成的数据一致
    torch.manual_seed(42)
    np.random.seed(42)
    
    sim = GKDDatasetSimulator(num_nodes=3000, num_workers=300, num_tasks=100)
    sim.generate_and_save()