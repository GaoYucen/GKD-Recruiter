import numpy as np
import networkx as nx
import time
import os
import heapq
from evaluate import GKDEvaluator

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

def run_ndd(G, q_matrix, worker_indices, K=50):
    """
    NDD (Node Degree Decay): 带衰减的度中心性贪心
    策略：选出预期影响力(出度权重和)最大的节点，随后对其邻居的影响力进行打折惩罚，避免重叠。
    """
    print("\n[算法] 运行 NDD (Node Degree Decay)...")
    start_time = time.time()
    
    # 初始化每个候选工人的预期出度影响力 (出边权重之和)
    scores = {w: sum(G[w][v].get('weight', 0.1) for v in G.successors(w)) for w in worker_indices}
    seed_pairs = []
    
    for _ in range(K):
        # 找出当前得分最高的节点
        if not scores: break
        best_w = max(scores, key=scores.get)
        
        # 分配质量潜力最高的任务
        w_idx = np.where(worker_indices == best_w)[0][0]
        best_task = int(np.argmax(q_matrix[w_idx]))
        seed_pairs.append((best_w, best_task))
        
        # 从候选池中移除
        scores.pop(best_w)
        
        # 核心机制：度数衰减 (打折其邻居的得分，因为该节点已被激活，其邻居被重复激活的边际价值降低)
        for neighbor in G.successors(best_w):
            if neighbor in scores:
                discount = G[best_w][neighbor].get('weight', 0.1)
                scores[neighbor] = max(0, scores[neighbor] - discount)
                
    print(f"⏱️ 推断耗时: {time.time() - start_time:.4f}s")
    return seed_pairs

def run_celf(G, q_matrix, a_matrix, task_demands, worker_indices, K=50, m=5):
    """
    CELF (Cost-Effective Lazy Forward): 延迟贪心影响力最大化
    策略：利用次模性假设（尽管当前环境是非次模的），通过优先队列减少边际收益的重复计算。
    """
    print(f"\n[算法] 运行 CELF (Cost-Effective Lazy Forward) ...")
    print(f"⚠️  警告: CELF 运行极其缓慢。为了可行性，已将每个工人的任务候选池裁剪为 Top-{m}。")
    start_time = time.time()
    
    # 使用较少的 MC 模拟次数来加速 CELF 的内部评估 (仅用于选种阶段)
    fast_evaluator = GKDEvaluator(G, q_matrix, a_matrix, task_demands, worker_indices, num_simulations=10)
    
    # 1. 构建精简候选池 (只考虑每个工人最擅长的前 m 个任务)
    candidates = []
    for row_idx, w in enumerate(worker_indices):
        # 综合兴趣和质量选出前 m 个任务
        scores = q_matrix[row_idx] * a_matrix[row_idx]
        top_tasks = np.argsort(scores)[-m:]
        for t in top_tasks:
            candidates.append((w, int(t)))
            
    print(f"   => 候选对总数: {len(candidates)}")
    
    # 2. 第一轮 (First Pass) - 计算所有候选对的初始边际收益
    print("   => 正在进行第 1 轮全量扫描 (最耗时的一步，请耐心等待...)")
    heap = []
    for i, (w, t) in enumerate(candidates):
        # 计算仅有这一个种子的 ETS
        ets = fast_evaluator.evaluate([(w, t)])['Effective_Task_Satisfaction']
        # Python 的 heapq 是最小堆，为了实现最大堆，我们将收益取负
        heapq.heappush(heap, (-ets, w, t, 0)) # 0 表示这是在第 0 个种子时计算的收益
        
        if (i + 1) % 300 == 0:
            print(f"      进度: {i + 1} / {len(candidates)}")

    # 3. 延迟贪心循环 (Lazy Forward Pass)
    seed_pairs = []
    current_ets = 0.0
    
    print("   => 正在执行延迟贪心增量选择...")
    while len(seed_pairs) < K and heap:
        neg_gain, w, t, iter_calc = heapq.heappop(heap)
        
        # 如果这个收益是在当前种子数量下计算的，说明它就是真实的当前最大边际收益
        if iter_calc == len(seed_pairs):
            seed_pairs.append((w, t))
            current_ets += (-neg_gain) # 近似累加 (实际有偏差因为非次模)
            print(f"      已选种子: {len(seed_pairs)}/{K} | 节点: {w}, 任务: {t}")
        else:
            # 否则，它可能是过时的。我们需要重新计算它在当前种子集合下的边际收益
            new_eval = fast_evaluator.evaluate(seed_pairs + [(w, t)])['Effective_Task_Satisfaction']
            current_true_ets = fast_evaluator.evaluate(seed_pairs)['Effective_Task_Satisfaction'] if seed_pairs else 0.0
            
            marginal_gain = new_eval - current_true_ets
            # 重新压入堆中，标记为当前轮次计算的
            heapq.heappush(heap, (-marginal_gain, w, t, len(seed_pairs)))
            
    print(f"⏱️ 推断耗时: {time.time() - start_time:.4f}s")
    return seed_pairs

if __name__ == "__main__":
    # 初始化环境与评估器 (评估最终结果时使用 50 次 MC 保证严谨)
    G, q_matrix, a_matrix, task_demands, worker_indices = load_env_data('data/env_params')
    evaluator = GKDEvaluator(G, q_matrix, a_matrix, task_demands, worker_indices, num_simulations=50)
    budget_K = 50 
    
    print(f"======== 测试设置: 预算 K = {budget_K} ========")

    # 1. 评估 NDD
    ndd_seeds = run_ndd(G, q_matrix, worker_indices, K=budget_K)
    ndd_results = evaluator.evaluate(ndd_seeds)
    print(f"📊 NDD 核心 ETS: {ndd_results['Effective_Task_Satisfaction']:.4f}")
    print(f"   传统传播范围 (节点数): {ndd_results['Expected_Influence_Spread']:.2f}")

    # 2. 评估 CELF (耗时较长)
    # 裁剪参数 m=5，以保证几分钟内能跑出结果。如果您想感受原汁原味的绝望，可以把 m 调到 100。
    celf_seeds = run_celf(G, q_matrix, a_matrix, task_demands, worker_indices, K=budget_K, m=5)
    celf_results = evaluator.evaluate(celf_seeds)
    print(f"📊 CELF 核心 ETS: {celf_results['Effective_Task_Satisfaction']:.4f}")
    print(f"   传统传播范围 (节点数): {celf_results['Expected_Influence_Spread']:.2f}")