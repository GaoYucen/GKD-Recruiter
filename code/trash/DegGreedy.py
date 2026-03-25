import networkx as nx
from ec_func import readGraph, influence_spread_ic

n_subarea = 100
n_users = 3000
n_seed = 5
nodefilename = 'data/Gowalla/input_node_3000_0.txt'
edgefilename = 'data/Gowalla/input_edge_3000_0.txt'

def select_seed_deggreedy(G, seed_num):
    """
    简单度贪心算法，选择度最大的若干节点作为种子
    """
    # 计算每个节点的度（后继节点数）
    node_degrees = [(node, len(list(G.neighbors(node)))) for node in G.nodes]
    # 按度从大到小排序
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    # 选择前seed_num个节点
    seeds = set([node for node, deg in node_degrees[:seed_num]])
    return seeds

def main():
    G = readGraph(nodefilename, edgefilename, n_subarea)
    seeds = select_seed_deggreedy(G, n_seed)
    print(f"Selected seed nodes (total {len(seeds)}):")
    print(sorted(seeds))
    avg_ec = influence_spread_ic(G, seeds, n_subarea, num_simulations=100)
    print(f"Estimated effective coverage (EC, 100 simulations): {avg_ec:.4f}")

if __name__ == '__main__':
    main()
