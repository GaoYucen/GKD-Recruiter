import networkx as nx
import random

def gen_graph(n_users=300, n_subarea=10, avg_deg=8, seed=42,
              nodefile='input_node_300_0.txt', edgefile='input_edge_300_0.txt'):
    random.seed(seed)
    # 生成有向BA图
    G = nx.barabasi_albert_graph(n_users, avg_deg // 2, seed=seed)
    G = G.to_directed()
    # 节点属性：每个节点每个子区域的任务质量，均匀分布[0.2, 1.0]
    with open(nodefile, 'w') as nf:
        for node in G.nodes:
            weights = [f"{random.uniform(0.2, 1.0):.3f}" for _ in range(n_subarea)]
            nf.write(f"{node}\t" + "\t".join(weights) + "\n")
    # 边属性：激活概率，均匀分布[0.01, 0.1]
    with open(edgefile, 'w') as ef:
        for u, v in G.edges:
            prob = random.uniform(0.01, 0.1)
            ef.write(f"{u}\t{v}\t{prob:.4f}\n")

def gen_celf_advantage_graph(n_users=300, n_subarea=10, avg_deg=6, seed=42,
                             nodefile='input_node_300_0.txt', edgefile='input_edge_300_0.txt'):
    random.seed(seed)
    # 生成分社区结构
    G = nx.connected_caveman_graph(5, n_users // 5)
    G = G.to_directed()
    # 每个社区分配不同的高质量subarea
    community_subareas = [list(range(i*2, (i+1)*2)) for i in range(5)]
    with open(nodefile, 'w') as nf:
        for node in G.nodes:
            # 识别社区
            comm = node // (n_users // 5)
            weights = []
            for j in range(n_subarea):
                if j in community_subareas[comm]:
                    weights.append(f"{random.uniform(0.8, 1.0):.3f}")
                else:
                    weights.append(f"{random.uniform(0.1, 0.3):.3f}")
            nf.write(f"{node}\t" + "\t".join(weights) + "\n")
    # 边权
    with open(edgefile, 'w') as ef:
        for u, v in G.edges:
            # 社区内边权高，社区间边权低
            if u // (n_users // 5) == v // (n_users // 5):
                prob = random.uniform(0.05, 0.15)
            else:
                prob = random.uniform(0.01, 0.03)
            ef.write(f"{u}\t{v}\t{prob:.4f}\n")

if __name__ == '__main__':
    # 任选其一调用
    # gen_graph(
    #     n_users=3000,
    #     n_subarea=100,
    #     avg_deg=8,
    #     seed=42,
    #     nodefile='data/input_node_3000_0.txt',
    #     edgefile='data/input_edge_3000_0.txt'
    # )
    gen_celf_advantage_graph(
        n_users=3000,
        n_subarea=100,
        avg_deg=6,
        seed=42,
        nodefile='data/input_node_3000_0.txt',
        edgefile='data/input_edge_3000_0.txt'
    )
