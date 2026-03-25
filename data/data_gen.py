import torch
import networkx as nx
import numpy as np
import pickle

class GKDDatasetSimulator:
    def __init__(self, num_nodes=3000, num_workers=300, num_tasks=100, embed_dim=64):
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        
    def generate_data(self):
        print(f"Generating synthetic data: |V|={self.num_nodes}, |U|={self.num_workers}, |T|={self.num_tasks}")
        
        # 1. 社交网络图 (Social Graph) - 保持幂律分布
        # 使用 Barabasi-Albert 模型生成无标度网络
        G = nx.barabasi_albert_graph(self.num_nodes, m=3)
        # 转换为有向图以支持 IGAT 的双向信息流
        G = G.to_directed()
        
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # # 随机生成社交影响力的基础概率 w_ij
        # edge_weight = torch.rand(edge_index.size(1), dtype=torch.float)
        # 模拟真实世界中极低的社交转化率 (0.01 到 0.1 之间)
        edge_weight = torch.rand(edge_index.size(1), dtype=torch.float) * 0.09 + 0.01
        
        # 2. 节点初始特征 (Initial Features)
        # 在实际中这里可能是用户画像/历史轨迹的 MLP 降维结果，这里用正态分布模拟
        node_features = torch.randn(self.num_nodes, self.embed_dim)
        task_features = torch.randn(self.num_tasks, self.embed_dim)
        
        # 3. 随机选取工人 (Workers)
        worker_indices = torch.randperm(self.num_nodes)[:self.num_workers]
        worker_features = node_features[worker_indices]
        
        # 4. 生成空间位置与任务属性 (Spatial & Task Attributes)
        # 模拟在 10x10 的空间网格内的 2D 坐标
        worker_locations = torch.rand(self.num_workers, 2) * 10 
        task_locations = torch.rand(self.num_tasks, 2) * 10
        
        # 任务报酬 (Rewards) 和 任务需求量 (Demands)
        task_rewards = torch.rand(self.num_tasks) * 10 + 5 # 5 到 15 之间
        # task_demands = torch.randint(1, 5, (self.num_tasks,), dtype=torch.float)
        # 提高任务的吞吐容量，避免几个人就饱和
        task_demands = torch.randint(10, 50, (self.num_tasks,), dtype=torch.float)
        
        # 5. 计算 Quality Potential (q) 和 Task Affinity (a) (Eq. 12)
        # 计算所有工人到任务的距离矩阵
        dist_matrix = torch.cdist(worker_locations, task_locations)
        D_max = dist_matrix.max()
        
        # Quality Potential: q = 1 - dis / D_max
        q_matrix = 1.0 - (dist_matrix / D_max)
        
        # 模拟历史访问频次 n(v_i, t_l)
        visit_freq = torch.randint(0, 10, (self.num_workers, self.num_tasks), dtype=torch.float)
        N_max = visit_freq.max()
        
        # # Task Affinity: a = r_l * (n / N_max)
        # a_matrix = task_rewards.unsqueeze(0) * (visit_freq / N_max)

        # 原始计算
        raw_a_matrix = task_rewards.unsqueeze(0) * (visit_freq / N_max)

        # 【新增】将 a_matrix 归一化到 [0, 1] 区间
        # 这样保证 p_ij = w_ij * a_jl 永远小于等于纯社交概率 w_ij，符合线下任务更难拉新的现实
        a_matrix = raw_a_matrix / raw_a_matrix.max()
        
        # 6. 为 RGCN 构建修剪后的异构边 (Top-m Tasks, m=5)
        m = 5
        top_m_vals, top_m_indices = torch.topk(q_matrix, k=m, dim=1)
        
        wt_edge_src = torch.arange(self.num_workers).view(-1, 1).repeat(1, m).view(-1)
        wt_edge_dst = top_m_indices.view(-1)
        
        # 构建异构图的 edge_index
        # 关系 0: worker -> task, 关系 1: task -> worker
        wt_edge_index = torch.stack([wt_edge_src, wt_edge_dst], dim=0)
        tw_edge_index = torch.stack([wt_edge_dst, wt_edge_src], dim=0)
        
        hetero_edge_index = torch.cat([wt_edge_index, tw_edge_index], dim=1)
        hetero_edge_type = torch.cat([
            torch.zeros(wt_edge_index.size(1), dtype=torch.long),
            torch.ones(tw_edge_index.size(1), dtype=torch.long)
        ])

        # 7. Correlation Layer 的相似度矩阵 beta_ij
        # 使用余弦相似度模拟
        worker_sim_adj = torch.nn.functional.cosine_similarity(worker_features.unsqueeze(1), worker_features.unsqueeze(0), dim=2)
        task_sim_adj = torch.nn.functional.cosine_similarity(task_features.unsqueeze(1), task_features.unsqueeze(0), dim=2)

        return {
            "social_graph": (node_features, edge_index, edge_weight),
            "hetero_graph": (worker_features, task_features, hetero_edge_index, hetero_edge_type),
            "correlation": (worker_sim_adj, task_sim_adj),
            "metrics": {"q_matrix": q_matrix, "a_matrix": a_matrix, "demands": task_demands, "rewards": task_rewards},
            "worker_indices": worker_indices
        }

# 测试一下
if __name__ == "__main__":
    sim = GKDDatasetSimulator(num_nodes=3000)
    data = sim.generate_data()
    print("Dataset generated successfully!")
    print(f"Heterogeneous Edge Index Shape: {data['hetero_graph'][2].shape}")
    
    # 保存数据到 .txt 文件
    import os
    if not os.path.exists('data/sample'):
        os.makedirs('data/sample')
    
    # social_graph: (node_features, edge_index, edge_weight)
    node_features, edge_index, edge_weight = data['social_graph']
    np.savetxt('data/sample/node_features.txt', node_features.numpy())
    np.savetxt('data/sample/edge_index.txt', edge_index.numpy().T, fmt='%d')  # 转置保存
    np.savetxt('data/sample/edge_weight.txt', edge_weight.numpy())
    
    # hetero_graph: (worker_features, task_features, hetero_edge_index, hetero_edge_type)
    worker_features, task_features, hetero_edge_index, hetero_edge_type = data['hetero_graph']
    np.savetxt('data/sample/worker_features.txt', worker_features.numpy())
    np.savetxt('data/sample/task_features.txt', task_features.numpy())
    np.savetxt('data/sample/hetero_edge_index.txt', hetero_edge_index.numpy().T, fmt='%d')
    np.savetxt('data/sample/hetero_edge_type.txt', hetero_edge_type.numpy(), fmt='%d')
    
    # correlation: (worker_sim_adj, task_sim_adj)
    worker_sim_adj, task_sim_adj = data['correlation']
    np.savetxt('data/sample/worker_sim_adj.txt', worker_sim_adj.numpy())
    np.savetxt('data/sample/task_sim_adj.txt', task_sim_adj.numpy())
    
    # metrics: {"q_matrix": q_matrix, "a_matrix": a_matrix, "demands": task_demands, "rewards": task_rewards}
    metrics = data['metrics']
    np.savetxt('data/sample/q_matrix.txt', metrics['q_matrix'].numpy())
    np.savetxt('data/sample/a_matrix.txt', metrics['a_matrix'].numpy())
    np.savetxt('data/sample/task_demands.txt', metrics['demands'].numpy())
    np.savetxt('data/sample/task_rewards.txt', metrics['rewards'].numpy())
    
    # worker_indices
    np.savetxt('data/sample/worker_indices.txt', data['worker_indices'].numpy(), fmt='%d')
    
    print("Data saved to data/sample/ directory as .txt files")