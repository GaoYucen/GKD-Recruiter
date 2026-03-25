import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import networkx as nx
import time
import os
from models.evaluate import GKDEvaluator

def load_env_data(env_dir='data/env_params'):
    """加载评估所需的环境真理数据"""
    print(f"📂 正在加载环境数据 ({env_dir}/)...")
    edge_index = np.loadtxt(os.path.join(env_dir, 'edge_index.txt'), dtype=int)
    w_ij = np.loadtxt(os.path.join(env_dir, 'w_ij.txt'), dtype=float)
    
    G = nx.DiGraph()
    edges_to_add = [
        (edge_index[i][0], edge_index[i][1], {'weight': w_ij[i]}) 
        for i in range(len(w_ij))
    ]
    G.add_edges_from(edges_to_add)

    q_matrix = np.loadtxt(os.path.join(env_dir, 'q_matrix.txt'), dtype=float)
    a_matrix = np.loadtxt(os.path.join(env_dir, 'a_matrix.txt'), dtype=float)
    task_demands = np.loadtxt(os.path.join(env_dir, 'task_demands.txt'), dtype=float)
    worker_indices = np.loadtxt(os.path.join(env_dir, 'worker_indices.txt'), dtype=int)
    
    return G, q_matrix, a_matrix, task_demands, worker_indices

def build_full_matrices(q_matrix, a_matrix, worker_indices, num_nodes):
    """将候选工人的局部特征广播到全图，供 ComGreedy 计算全局收益使用"""
    num_tasks = q_matrix.shape[1]
    full_q = np.zeros((num_nodes, num_tasks))
    full_a = np.zeros((num_nodes, num_tasks))
    
    # 用均值填充普通用户的属性
    full_q[:] = np.mean(q_matrix, axis=0)
    full_a[:] = np.mean(a_matrix, axis=0)
    
    for row_idx, node_id in enumerate(worker_indices):
        full_q[node_id] = q_matrix[row_idx]
        full_a[node_id] = a_matrix[row_idx]
        
    return full_q, full_a

def run_deg_greedy(G, q_matrix, worker_indices, K=50):
    """
    DegGreedy: 纯拓扑中心性贪心
    策略：挑选出度（影响力）最大的前 K 个工人，并将其分配给他们最擅长的任务
    """
    print("\n[算法] 运行 DegGreedy (度中心性贪心)...")
    start_time = time.time()
    
    worker_degrees = {w: G.out_degree(w) for w in worker_indices}
    # 按出度降序排列
    sorted_workers = sorted(worker_degrees.keys(), key=lambda x: worker_degrees[x], reverse=True)
    
    seed_pairs = []
    for w in sorted_workers:
        # 找到该工人在矩阵中的行索引
        w_idx = np.where(worker_indices == w)[0][0]
        # 为他分配质量潜力最高的任务
        best_task = int(np.argmax(q_matrix[w_idx]))
        seed_pairs.append((w, best_task))
        if len(seed_pairs) >= K:
            break
            
    print(f"⏱️ 推断耗时: {time.time() - start_time:.4f}s")
    return seed_pairs

def run_com_greedy(G, full_q, full_a, worker_indices, num_tasks, K=50):
    """
    ComGreedy: 综合指标贪心 (论文中最强的启发式 Baseline)
    策略：计算每个 (Worker, Task) 对在单步扩散下的期望质量收益，选取 Top-K
    """
    print("\n[算法] 运行 ComGreedy (综合期望贪心)...")
    start_time = time.time()
    
    pair_scores = []
    # 遍历所有可能的 (工人, 任务) 组合
    for w in worker_indices:
        for t in range(num_tasks):
            score = 0.0
            # 论文启发式公式: sum( w_ij * a_j^l * q_j^l )
            for neighbor in G.successors(w):
                w_ij = G[w][neighbor].get('weight', 0.1)
                score += w_ij * full_a[neighbor, t] * full_q[neighbor, t]
            pair_scores.append((score, w, t))
            
    # 按综合得分降序排列
    pair_scores.sort(key=lambda x: x[0], reverse=True)
    
    seed_pairs = [(int(w), int(t)) for score, w, t in pair_scores[:K]]
    print(f"⏱️ 推断耗时: {time.time() - start_time:.4f}s")
    return seed_pairs

if __name__ == "__main__":
    # 1. 加载数据与初始化
    G, q_matrix, a_matrix, task_demands, worker_indices = load_env_data('data/env_params')
    num_nodes = G.number_of_nodes()
    num_tasks = len(task_demands)
    full_q, full_a = build_full_matrices(q_matrix, a_matrix, worker_indices, num_nodes)
    
    evaluator = GKDEvaluator(G, q_matrix, a_matrix, task_demands, worker_indices, num_simulations=50)
    
    budget_K = 50 # 设定论文实验中的标准预算
    print(f"======== 测试设置: 预算 K = {budget_K} ========")

    # 2. 评估 DegGreedy
    deg_seeds = run_deg_greedy(G, q_matrix, worker_indices, K=budget_K)
    deg_results = evaluator.evaluate(deg_seeds)
    print(f"📊 DegGreedy 核心 ETS: {deg_results['Effective_Task_Satisfaction']:.4f}")
    print(f"   传统传播范围 (节点数): {deg_results['Expected_Influence_Spread']:.2f}")

    # 3. 评估 ComGreedy
    com_seeds = run_com_greedy(G, full_q, full_a, worker_indices, num_tasks, K=budget_K)
    com_results = evaluator.evaluate(com_seeds)
    print(f"📊 ComGreedy 核心 ETS: {com_results['Effective_Task_Satisfaction']:.4f}")
    print(f"   传统传播范围 (节点数): {com_results['Expected_Influence_Spread']:.2f}")