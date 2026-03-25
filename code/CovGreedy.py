import networkx as nx
import random
from ec_func import readGraph, influence_spread_ic

n_subarea = 100
n_users = 3000
n_seed = 5
nodefilename = 'data/Gowalla/input_node_3000_0.txt'
edgefilename = 'data/Gowalla/input_edge_3000_0.txt'

def select_seed_covgreedy(G, seed_num, n_subarea):
    """
    覆盖贪心算法选种子节点
    计算每个节点的覆盖度，选覆盖度最大的若干节点作为种子
    """
    # 计算每个节点的覆盖度（所有subarea的任务质量之和）
    node_cover = []
    for node in G.nodes:
        cover = sum(G.nodes[node]['weight'])
        node_cover.append((cover, node))
    # 按覆盖度从大到小排序，选前seed_num个
    node_cover.sort(reverse=True)
    seeds = set(node for _, node in node_cover[:seed_num])
    return seeds

def main():
    G = readGraph(nodefilename, edgefilename, n_subarea)
    seeds = select_seed_covgreedy(G, n_seed, n_subarea)
    print(f"Selected seed nodes (total {len(seeds)}):")
    print(sorted(seeds))
    avg_spread = influence_spread_ic(G, seeds, n_subarea, num_simulations=100)
    print(f"Estimated influence spread (IC model, 100 simulations): {avg_spread:.2f}")

if __name__ == '__main__':
    main()