import numpy as np
import networkx as nx
import random
from typing import List, Tuple, Dict, Set

class GKDEvaluator:
    def __init__(self, social_graph: nx.DiGraph, q_matrix: np.ndarray, a_matrix: np.ndarray, 
                 task_demands: np.ndarray, worker_indices: np.ndarray, num_simulations: int = 100):
        self.G = social_graph
        self.task_demands = task_demands
        self.num_simulations = num_simulations
        
        self.num_nodes = social_graph.number_of_nodes() # 3000
        self.num_tasks = q_matrix.shape[1] # 100
        
        # ---------------------------------------------------------
        # 【核心修复】：将 300x100 的局部矩阵对齐到 3000 个全局节点
        # ---------------------------------------------------------
        self.full_q_matrix = np.zeros((self.num_nodes, self.num_tasks))
        self.full_a_matrix = np.zeros((self.num_nodes, self.num_tasks))
        
        # 1. 计算 300 个候选工人的均值，用来作为 2700 个普通大众的默认属性
        # （假设普通社交用户被影响后，其参与意愿和质量服从平均水平）
        avg_q = np.mean(q_matrix, axis=0)
        avg_a = np.mean(a_matrix, axis=0)
        
        self.full_q_matrix[:] = avg_q
        self.full_a_matrix[:] = avg_a
        
        # 2. 将 300 个候选工人的专属真实数据，覆盖到他们对应的节点 ID 上
        for row_idx, node_id in enumerate(worker_indices):
            self.full_q_matrix[node_id] = q_matrix[row_idx]
            self.full_a_matrix[node_id] = a_matrix[row_idx]

    def evaluate(self, seed_pairs: List[Tuple[int, int]]) -> Dict[str, float]:
        task_seeds = {l: set() for l in range(self.num_tasks)}
        unique_seed_users = set()
        
        for worker_id, task_id in seed_pairs:
            task_seeds[task_id].add(worker_id)
            unique_seed_users.add(worker_id)

        task_ets_list = []
        cumulative_qualities = []

        for task_id in range(self.num_tasks):
            seeds = task_seeds[task_id]
            if not seeds:
                task_ets_list.append(0.0)
                cumulative_qualities.append(0.0)
                continue

            expected_quality = self._simulate_task_aware_ic(task_id, seeds)
            
            # 论文公式 (4): 满足需求即截断
            demand = self.task_demands[task_id]
            ets = min(expected_quality / demand, 1.0)

            task_ets_list.append(ets)
            cumulative_qualities.append(expected_quality)

        mean_ets = np.mean(task_ets_list)
        mean_cumulative_quality = np.mean(cumulative_qualities)
        expected_spread = self._simulate_standard_ic(unique_seed_users)

        return {
            "Effective_Task_Satisfaction": float(mean_ets),
            "Mean_Cumulative_Quality": float(mean_cumulative_quality),
            "Expected_Influence_Spread": float(expected_spread),
            "Seed_Set_Size": len(seed_pairs),
            "Unique_Seed_Users": len(unique_seed_users)
        }

    def _simulate_task_aware_ic(self, task_id: int, seeds: Set[int]) -> float:
        total_quality = 0.0

        for _ in range(self.num_simulations):
            activated = set(seeds)
            newly_activated = set(seeds)

            while newly_activated:
                next_activated = set()
                for node in newly_activated:
                    for neighbor in self.G.successors(node): 
                        if neighbor not in activated:
                            w_ij = self.G[node][neighbor].get('weight', 0.1)
                            # 使用扩充后的 full_a_matrix，确保不会越界
                            a_jl = self.full_a_matrix[neighbor, task_id]
                            p_ij_t = w_ij * a_jl

                            if random.random() < p_ij_t:
                                next_activated.add(neighbor)

                newly_activated = next_activated
                activated.update(newly_activated)

            # 使用扩充后的 full_q_matrix
            sim_quality = sum(self.full_q_matrix[node, task_id] for node in activated)
            total_quality += sim_quality

        return total_quality / self.num_simulations

    def _simulate_standard_ic(self, seeds: Set[int]) -> float:
        if not seeds:
            return 0.0

        total_activated = 0
        
        for _ in range(self.num_simulations):
            activated = set(seeds)
            newly_activated = set(seeds)

            while newly_activated:
                next_activated = set()
                for node in newly_activated:
                    for neighbor in self.G.successors(node):
                        if neighbor not in activated:
                            w_ij = self.G[node][neighbor].get('weight', 0.1)
                            if random.random() < w_ij:
                                next_activated.add(neighbor)
                                
                newly_activated = next_activated
                activated.update(newly_activated)

            total_activated += len(activated)

        return total_activated / self.num_simulations