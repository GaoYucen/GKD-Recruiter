import numpy as np
import networkx as nx
import random
import os
import time

# 导入我们刚刚编写的评估器
# 确保 evaluate.py 在同级目录下
from evaluate import GKDEvaluator

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



def load_real_gowalla_data(node_file, edge_file, num_tasks=100, num_workers=300):
    """
    读取真实的 Gowalla 数据，并转化为 GKD-Recruiter 评估器所需的矩阵格式
    """
    print(f"正在加载真实 Gowalla 数据...")
    
    G = nx.DiGraph()
    # 1. 加载真实的社交边和影响概率 (w_ij)
    with open(edge_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            G.add_edge(u, v, weight=w)
            
    num_nodes = G.number_of_nodes()
    print(f"真实社交图构建完成: {num_nodes} 节点, {G.number_of_edges()} 边")

    # 2. 加载真实的节点在各子区域(任务)的权重
    raw_node_matrix = np.zeros((num_nodes, num_tasks))
    with open(node_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            # 取出该节点在 100 个 subarea 的原始特征值 (可能是历史签到频次)
            weights = [float(x) for x in parts[1:1+num_tasks]]
            raw_node_matrix[node_id] = weights

    # ==========================================
    # 核心映射：将单维的真实权重，拆解为您的双重属性，并严格归一化
    # ==========================================
    
    # 方案 A: 假设原始数据代表“签到频次”，我们将其归一化后作为“任务亲和力 (a_matrix)”
    # 除以最大值，确保 a_jl 严格在 [0, 1] 之间，防止概率爆炸！
    a_matrix = raw_node_matrix / (raw_node_matrix.max() + 1e-9)
    
    # 方案 B: 质量潜力 (q_matrix) 
    # 在真实世界中，去得越多的人通常越熟悉该区域，质量潜力也越高。
    # 我们可以在 a_matrix 基础上加一点随机噪声模拟 q_matrix，或者直接复用归一化后的特征
    q_matrix = a_matrix.copy() * 0.8 + np.random.rand(num_nodes, num_tasks) * 0.2
    # 确保 q_matrix 也在 [0, 1] 范围内
    q_matrix = np.clip(q_matrix, 0.0, 1.0)

    # 3. 按照论文设定：挑选前 300 名活跃用户作为“候选工人池” (Worker Pool)
    # 按节点在所有任务上的总活跃度排序
    node_activity = raw_node_matrix.sum(axis=1)
    worker_indices = np.argsort(node_activity)[-num_workers:]

    # 4. 生成任务需求 (Demands)
    # 论文中提到 demand 是基于轨迹密度决定的。这里我们将其设为真实轨迹特征总和的一部分
    task_activity_sum = raw_node_matrix.sum(axis=0)
    # 映射到合理的区间，比如需要 5 到 30 个有效贡献
    task_demands = 5.0 + (task_activity_sum / task_activity_sum.max()) * 25.0

    print("矩阵对齐完成且已归一化！")
    return G, q_matrix, a_matrix, task_demands, worker_indices

# --------- 使用示例 ---------
if __name__ == "__main__":
    node_path = "data/input_node_3000_0.txt"
    edge_path = "data/input_edge_3000_0.txt"
    
    G, q_matrix, a_matrix, task_demands, worker_indices = load_real_gowalla_data(node_path, edge_path)
    
    # 接下来直接把这 5 个变量喂给您的 GKDEvaluator 即可！
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