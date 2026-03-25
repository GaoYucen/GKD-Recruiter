import numpy as np
import networkx as nx
import random
import os
import time

# 导入评估器
from evaluate import GKDEvaluator

def load_env_data(env_dir='data/env_params'):
    """
    从 env_params 目录加载评估所需的全部环境参数
    """
    print(f"正在从 '{env_dir}/' 目录加载环境真理数据...")
    
    # 1. 加载社交拓扑与传播概率 (w_ij)
    # edge_index 形状为 (E, 2)，w_ij 形状为 (E,)
    edge_index = np.loadtxt(os.path.join(env_dir, 'edge_index.txt'), dtype=int)
    w_ij = np.loadtxt(os.path.join(env_dir, 'w_ij.txt'), dtype=float)
    
    # 构建 NetworkX 有向图
    G = nx.DiGraph()
    # 这里的 i 指向 edge_index 的第 i 行，对应 w_ij 的第 i 个概率值
    edges_to_add = [
        (edge_index[i][0], edge_index[i][1], {'weight': w_ij[i]}) 
        for i in range(len(w_ij))
    ]
    G.add_edges_from(edges_to_add)
    print(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边 (已关联 w_ij)")

    # 2. 加载核心矩阵 (由论文公式推导产生)
    q_matrix = np.loadtxt(os.path.join(env_dir, 'q_matrix.txt'), dtype=float)
    a_matrix = np.loadtxt(os.path.join(env_dir, 'a_matrix.txt'), dtype=float)
    task_demands = np.loadtxt(os.path.join(env_dir, 'task_demands.txt'), dtype=float)
    worker_indices = np.loadtxt(os.path.join(env_dir, 'worker_indices.txt'), dtype=int)
    
    print(f"参数加载完成: q_matrix {q_matrix.shape}, a_matrix {a_matrix.shape}, tasks={len(task_demands)}")
    
    return G, q_matrix, a_matrix, task_demands, worker_indices

def mock_rl_agent_selection(worker_indices, num_tasks, budget=5):
    """
    模拟算法选出的种子对 (Worker ID, Task ID)
    """
    selected_seeds = []
    # 随机挑选 budget 数量的种子对作为演示
    # 这里的 Worker ID 必须是 worker_indices 中的真实节点 ID
    for _ in range(budget):
        w = random.choice(worker_indices) 
        t = random.randint(0, num_tasks - 1) 
        if (w, t) not in selected_seeds:
            selected_seeds.append((w, t))
    return selected_seeds

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    # 路径检查
    env_path = 'data/env_params'
    if not os.path.exists(env_path):
        print(f"错误: 找不到 '{env_path}' 目录！请先运行 data_gen_2.py。")
        exit(1)

    # 1. 加载环境参数
    G, q_matrix, a_matrix, task_demands, worker_indices = load_env_data(env_path)
    num_tasks = len(task_demands)

    # 2. 模拟产出种子对 (例如预算 K=5)
    seed_pairs = mock_rl_agent_selection(worker_indices, num_tasks, budget=5)
    print(f"\n模拟算法选出的 {len(seed_pairs)} 个种子对 (Worker ID, Task ID):")
    print(seed_pairs)

    # 3. 执行评估
    print(f"\n开始进行蒙特卡洛模拟评估 (num_simulations=50)...")
    start_time = time.time()
    
    evaluator = GKDEvaluator(
        social_graph=G, 
        q_matrix=q_matrix, 
        a_matrix=a_matrix, 
        task_demands=task_demands, 
        worker_indices=worker_indices,
        num_simulations=50
    )

    results = evaluator.evaluate(seed_pairs)
    end_time = time.time()

    # 4. 打印报告
    print(f"\n{'='*40}")
    print(" 🎯 GKD-Recruiter 评估报告")
    print(f"{'='*40}")
    print(f"⏱️ 耗时: {end_time - start_time:.2f}s")
    print(f"👤 种子覆盖用户数: {results['Unique_Seed_Users']}")
    print("-" * 40)
    print(f"⭐ [核心指标] ETS (有效任务满意度): {results['Effective_Task_Satisfaction']:.4f}")
    print(f"📊 [过程指标] 平均累积质量 (C): {results['Mean_Cumulative_Quality']:.4f}")
    print(f"🌐 [基准指标] 期望影响力扩散范围: {results['Expected_Influence_Spread']:.2f} 节点")
    print(f"{'='*40}")