import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import networkx as nx
import time
import os
import heapq
from models.evaluate import GKDEvaluator

def load_env_data(env_dir='data/env_params'):
    """Load ground truth data for environmental evaluation."""
    print(f"📂 Loading environment data from ({env_dir}/)...")
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
    NDD (Node Degree Decay): Degree Centrality Greedy with Decay
    Strategy: Select the node with the highest expected influence (sum of out-edge weights), 
    then penalize the influence of its neighbors to avoid redundancy.
    """
    print("\n[Baseline] Running NDD (Node Degree Decay)...")
    start_time = time.time()
    
    # Initialize the expected out-degree influence (sum of out-edge weights) for each candidate worker
    scores = {w: sum(G[w][v].get('weight', 0.1) for v in G.successors(w)) for w in worker_indices}
    seed_pairs = []
    
    for _ in range(K):
        # Find the node with the current highest score
        if not scores: break
        best_w = max(scores, key=scores.get)
        
        # Assign the task with the highest quality potential
        w_idx = np.where(worker_indices == best_w)[0][0]
        best_task = int(np.argmax(q_matrix[w_idx]))
        seed_pairs.append((best_w, best_task))
        
        # Remove from the candidate pool
        scores.pop(best_w)
        
        # Core mechanism: Degree Decay (discount its neighbors' scores since this node is now activated, 
        # lowering the marginal value of duplicate neighbor activation)
        for neighbor in G.successors(best_w):
            if neighbor in scores:
                discount = G[best_w][neighbor].get('weight', 0.1)
                scores[neighbor] = max(0, scores[neighbor] - discount)
                
    print(f"⏱️ Inference Time: {time.time() - start_time:.4f}s")
    return seed_pairs

def run_celf(G, q_matrix, a_matrix, task_demands, worker_indices, K=50, m=5):
    """
    CELF (Cost-Effective Lazy Forward): Lazy Greedy Influence Maximization
    Strategy: Utilize submodularity assumptions (despite the non-submodular environment) 
    to reduce redundant marginal gain calculations via a priority queue.
    """
    print(f"\n[Baseline] Running CELF (Cost-Effective Lazy Forward) ...")
    print(f"⚠️ Warning: CELF is extremely slow. For feasibility, each worker's task candidate pool is trimmed to Top-{m}.")
    start_time = time.time()
    
    # Use fewer MC simulations to accelerate internal CELF evaluation (only for seeding phase)
    fast_evaluator = GKDEvaluator(G, q_matrix, a_matrix, task_demands, worker_indices, num_simulations=10)
    
    # 1. Build a refined candidate pool (only consider the top m tasks for each worker)
    candidates = []
    for row_idx, w in enumerate(worker_indices):
        # Select top m tasks based on composite interest and quality
        scores = q_matrix[row_idx] * a_matrix[row_idx]
        top_tasks = np.argsort(scores)[-m:]
        for t in top_tasks:
            candidates.append((w, int(t)))
            
    print(f"   => Total candidate pairs: {len(candidates)}")
    
    # 2. First Pass: Calculate initial marginal gain for all candidate pairs
    print("   => Performing Round 1 full scan (most time-consuming, please wait...)")
    heap = []
    for i, (w, t) in enumerate(candidates):
        # Calculate ETS for this single seed only
        ets = fast_evaluator.evaluate([(w, t)])['Effective_Task_Satisfaction']
        # Python's heapq is a min-heap; to implement a max-heap, we negate the gain
        heapq.heappush(heap, (-ets, w, t, 0)) # 0 indicates this gain was calculated with 0 prior seeds
        
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