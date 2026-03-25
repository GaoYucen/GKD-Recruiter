import random
from copy import deepcopy
import time

def read_graph_from_txt(file_path):
    """
    从边文件读取图数据，构建字典格式的图
    :param file_path: 边文件路径（格式：源节点\t目标节点\t权重）
    :return: 图字典（g），节点质量字典（预留）
    """
    g = {}  # 图结构：g[源节点][目标节点] = 传播概率
    quality = {}  # 节点质量（可扩展用于加权影响力计算）
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().replace('\t', ' ')
            parts = line.split()
            if len(parts) != 3:
                continue  # 跳过格式错误的行
            
            u = int(parts[0])
            v = int(parts[1])
            p = float(parts[2])  # 边的传播概率
            
            # 初始化节点
            if u not in g:
                g[u] = {}
            if v not in g:
                g[v] = {}
            
            # 添加有向边（独立级联模型通常为有向图）
            g[u][v] = p

    # 计算节点平均度
    for node in g:
        quality[node] = len(g[node])  # 简单以出度作为质量指标（可根据需求调整）
    
    # 打印平均度
    avg_degree = sum(quality.values()) / len(quality) if quality else 0
    print(f"Average node degree: {avg_degree:.2f}")

    
    return g, quality

def ic_influence_spread(g, seed_set, num_simulations=100):
    """
    基于独立级联模型计算种子集的影响力传播范围
    :param g: 图字典（节点间传播概率）
    :param seed_set: 种子节点集合（set类型）
    :param num_simulations: 蒙特卡洛模拟次数（默认100次）
    :return: 平均影响力传播范围，标准差
    """
    total_spread = []
    
    for _ in range(num_simulations):
        # 每次模拟初始化激活节点集合
        activated = set(seed_set)
        # 当前轮次待传播的节点队列
        current = set(seed_set)
        
        while current:
            next_nodes = set()
            for u in current:
                # 遍历当前节点的所有邻居
                for v, p in g[u].items():
                    # 邻居未被激活且满足传播概率
                    if v not in activated and random.random() < p:
                        activated.add(v)
                        next_nodes.add(v)
            # 下一轮传播节点更新
            current = next_nodes
        
        # 记录本次模拟的传播范围
        total_spread.append(len(activated))
    
    # 计算平均值和标准差
    avg_spread = sum(total_spread) / num_simulations
    std_spread = (sum((s - avg_spread) **2 for s in total_spread) / num_simulations)** 0.5
    
    return avg_spread, std_spread

def main():
    # 配置参数
    graph_path = "data/Gowalla/input_edge_3000_0.txt"  # 图数据文件路径
    seed_set = {0, 1, 2}  # 示例种子节点集
    num_simulations = 100  # 模拟次数
    
    # 读取图数据
    start_time = time.time()
    g, quality = read_graph_from_txt(graph_path)
    print(f"图加载完成，节点数: {len(g)}, 耗时: {time.time() - start_time:.2f}秒")
    
    # 计算影响力传播
    start_time = time.time()
    avg_spread, std_spread = ic_influence_spread(g, seed_set, num_simulations)
    print(f"影响力计算完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 输出结果
    print(f"种子节点集: {seed_set}")
    print(f"平均影响力传播范围: {avg_spread:.2f} (±{std_spread:.2f})")
    print(f"模拟次数: {num_simulations}")

if __name__ == "__main__":
    main()