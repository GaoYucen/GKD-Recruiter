import networkx as nx
import random

def readGraph(nodefilename, edgefilename, n_subarea):
    G = nx.DiGraph()
    with open(nodefilename) as nodefile:
        for line in nodefile:
            parts = line.strip().split('\t')
            nodeId = int(parts[0])
            nodeWeight = [float(x) for x in parts[1:1+n_subarea]]
            G.add_node(nodeId, weight=nodeWeight)
    with open(edgefilename) as edgefile:
        for line in edgefile:
            parts = line.strip().split('\t')
            node1 = int(parts[0])
            node2 = int(parts[1])
            edgeWeight = float(parts[2])
            G.add_edge(node1, node2, weight=edgeWeight)
    return G

def influence_spread_ic(G, S, n_subarea, num_simulations=1000):
    """
    计算被激活节点在不同subarea上的任务质量均值
    G: 网络图
    S: 种子节点集合
    n_subarea: 子区域数量
    num_simulations: 模拟次数
    返回平均任务质量（所有subarea的均值）
    """
    total_quality = [0.0 for _ in range(n_subarea)]
    for _ in range(num_simulations):
        activated = set(S)
        newly_activated = set(S)
        while newly_activated:
            next_activated = set()
            for node in newly_activated:
                for neighbor in G.neighbors(node):
                    if neighbor not in activated:
                        prob = G[node][neighbor]['weight']
                        if random.random() < prob:
                            next_activated.add(neighbor)
            newly_activated = next_activated - activated
            activated.update(newly_activated)
        # 累加所有被激活节点在每个subarea上的任务质量
        for node in activated:
            for j in range(n_subarea):
                total_quality[j] += G.nodes[node]['weight'][j]
    # 计算所有模拟、所有subarea的平均任务质量
    avg_quality = sum(total_quality) / (num_simulations * n_subarea)
    return avg_quality
# filepath: code/CovGreedy.py