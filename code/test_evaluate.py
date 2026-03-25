import numpy as np
import networkx as nx
import random
import os
import time

# 导入我们刚刚编写的评估器
# 确保 evaluate.py 在同级目录下
from evaluate import GKDEvaluator

def load_graph_and_matrices(data_dir='data/sample'):
    """
    从 data_gen_2.py 生成的 txt 文件中加载图数据和特征矩阵
    """
    print(f"正在从 '{data_dir}/' 目录加载数据...")
    
    # 1. 加载边和权重
    # data_gen_2.py 中使用了 .T 转置保存，所以读取出来形状是 (num_edges, 2)
    edge_index = np.loadtxt(os.path.join(data_dir, 'edge_index.txt'), dtype=int)
    edge_weight = np.loadtxt(os.path.join(data_dir, 'edge_weight.txt'), dtype=float)
    
    # 构建 NetworkX 有向图
    G = nx.DiGraph()
    edges_to_add = [
        (edge_index[i][0], edge_index[i][1], {'weight': edge_weight[i]}) 
        for i in range(len(edge_weight))
    ]
    G.add_edges_from(edges_to_add)
    print(f"图构建完成: 包含 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边.")

    # 2. 加载评估所需的核心矩阵
    q_matrix = np.loadtxt(os.path.join(data_dir, 'q_matrix.txt'), dtype=float)
    a_matrix = np.loadtxt(os.path.join(data_dir, 'a_matrix.txt'), dtype=float)
    task_demands = np.loadtxt(os.path.join(data_dir, 'task_demands.txt'), dtype=float)
    worker_indices = np.loadtxt(os.path.join(data_dir, 'worker_indices.txt'), dtype=int)
    
    print(f"矩阵加载完成: q_matrix {q_matrix.shape}, a_matrix {a_matrix.shape}")
    
    return G, q_matrix, a_matrix, task_demands, worker_indices

def mock_rl_agent_selection(worker_indices, num_tasks, budget=20):
    """
    模拟强化学习智能体（或贪心算法）选出的种子对 (Worker, Task)
    """
    selected_seeds = []
    # 随机挑选预算数量的种子对作为测试
    for _ in range(budget):
        w = random.choice(worker_indices) # 只能从候选工人池中选
        t = random.randint(0, num_tasks - 1) # 任务ID (0 到 num_tasks-1)
        # 去重（防止选出完全一样的 pair）
        if (w, t) not in selected_seeds:
            selected_seeds.append((w, t))
    return selected_seeds

if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    random.seed(42)
    np.random.seed(42)
    
    # 检查数据目录是否存在
    if not os.path.exists('data/sample'):
        print("错误: 找不到 'data/sample' 文件夹！请先运行 data_gen_2.py 生成合成数据。")
        exit(1)

    # 1. 加载数据
    G, q_matrix, a_matrix, task_demands, worker_indices = load_graph_and_matrices(data_dir='data/sample')
    num_tasks = task_demands.shape[0]

    # 2. 模拟算法选出的种子对集合 (假设预算 K=25)
    budget_K = 5
    seed_pairs = mock_rl_agent_selection(worker_indices, num_tasks, budget=budget_K)
    print(f"\n模拟算法选出的 {len(seed_pairs)} 个种子对 (Worker ID, Task ID):")
    print(seed_pairs)

    # 3. 初始化评估器
    # 提示: num_simulations 设为 50 加快测试速度，论文最终出图建议设为 1000 
    print("\n开始进行蒙特卡洛模拟评估 (num_simulations=50)...")
    start_time = time.time()

    evaluator = GKDEvaluator(
        social_graph=G, 
        q_matrix=q_matrix, 
        a_matrix=a_matrix, 
        task_demands=task_demands, 
        worker_indices=worker_indices,  # <--- 新增这行，将索引传进去
        num_simulations=50
    )

    # 4. 执行评估
    results = evaluator.evaluate(seed_pairs)
    end_time = time.time()

    # 5. 打印对比结果
    print(f"\n{'='*40}")
    print(" 🎯 评 估 结 果 报 告 (Evaluation Report)")
    print(f"{'='*40}")
    print(f"⏱️ 评估耗时: {end_time - start_time:.2f} 秒")
    print(f"👥 消耗预算 (选定种子对数量): {results['Seed_Set_Size']}")
    print(f"👤 实际触发的独立用户数: {results['Unique_Seed_Users']}")
    print("-" * 40)
    print(f"⭐ [论文核心指标] ETS (有效任务满意度): {results['Effective_Task_Satisfaction']:.4f}")
    print(f"📊 [中间过程指标] 未截断平均累积质量 (C): {results['Mean_Cumulative_Quality']:.4f}")
    print(f"🌐 [传统 IM 指标] 纯社交网络期望影响扩散: {results['Expected_Influence_Spread']:.2f} 个节点")
    print(f"{'='*40}")