import networkx as nx
import heapq
import time
from ec_func import readGraph, influence_spread_ic

n_subarea = 100
n_users = 3000
n_seed = 5
nodefilename = 'data/Gowalla/input_node_3000_0.txt'
edgefilename = 'data/Gowalla/input_edge_3000_0.txt'

def celf_greedy(G, seed_num, n_subarea, num_simulations=100):
    """
    CELF算法，最大化 influence_spread_ic 返回的平均任务质量
    """
    S = set()
    node_list = list(G.nodes)
    heap = []
    print("初始化每个节点的单节点增益...")
    for idx, node in enumerate(node_list):
        gain = influence_spread_ic(G, {node}, n_subarea, num_simulations)
        heapq.heappush(heap, (-gain, node, 0))
        if (idx + 1) % 100 == 0 or idx == len(node_list) - 1:
            print(f"  已完成 {idx + 1} / {len(node_list)} 个节点的增益计算")
    selected = set()
    last_update = {node: 0 for node in node_list}
    cur_spread = 0.0
    for k in range(seed_num):
        print(f"\n[CELF] 选取第 {k+1} 个种子节点...")
        t0 = time.time()
        while True:
            neg_gain, node, prev_k = heapq.heappop(heap)
            if last_update[node] == k:
                S.add(node)
                cur_spread += -neg_gain
                selected.add(node)
                print(f"  选中节点 {node}，边际增益: {-neg_gain:.4f}")
                # 打印当前种子集合的EC
                cur_ec = influence_spread_ic(G, S, n_subarea, num_simulations=100)
                print(f"  当前前{k+1}个种子节点集合的EC: {cur_ec:.4f}")
                break
            print(f"  重新计算节点 {node} 的边际增益...")
            new_gain = influence_spread_ic(G, S | {node}, n_subarea, num_simulations) - \
                       influence_spread_ic(G, S, n_subarea, num_simulations)
            heapq.heappush(heap, (-new_gain, node, k))
            last_update[node] = k
        t1 = time.time()
        print(f"  第 {k+1} 个种子节点选取完成，用时 {t1-t0:.2f} 秒")
    return S

def main():
    G = readGraph(nodefilename, edgefilename, n_subarea)
    seeds = celf_greedy(G, n_seed, n_subarea, num_simulations=50)
    print(f"Selected seed nodes (total {len(seeds)}):")
    print(sorted(seeds))
    avg_ec = influence_spread_ic(G, seeds, n_subarea, num_simulations=100)
    print(f"Estimated effective coverage (EC, 100 simulations): {avg_ec:.4f}")

if __name__ == '__main__':
    main()
