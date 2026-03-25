import random
import time
from is_func import read_graph_from_txt, ic_influence_spread

def random_seed_selection(g, k):
    """
    从图g中随机选择k个种子节点
    :param g: 图字典
    :param k: 种子节点数量
    :return: 种子节点集合（set）
    """
    nodes = list(g.keys())
    return set(random.sample(nodes, k))

def main():
    graph_path = "data/Gowalla/input_edge_3000_0.txt"
    k = 5
    num_simulations = 100
    num_trials = 20  # 新增：随机选种子并评估的次数

    # 读取图
    start_time = time.time()
    g, _ = read_graph_from_txt(graph_path)
    print(f"图加载完成，节点数: {len(g)}, 耗时: {time.time() - start_time:.2f}秒")

    all_avg_spread = []
    all_std_spread = []

    for i in range(num_trials):
        seed_set = random_seed_selection(g, k)
        avg_spread, std_spread = ic_influence_spread(g, seed_set, num_simulations)
        all_avg_spread.append(avg_spread)
        all_std_spread.append(std_spread)
        print(f"第{i+1}次: 种子节点: {seed_set}, 平均影响力: {avg_spread:.2f} (±{std_spread:.2f})")

    # 统计总体均值和标准差
    final_avg = sum(all_avg_spread) / num_trials
    final_std = (sum((x - final_avg) ** 2 for x in all_avg_spread) / num_trials) ** 0.5

    print(f"\n{num_trials}次实验的平均影响力: {final_avg:.2f} (±{final_std:.2f})")
    print(f"每次IC模拟次数: {num_simulations}")

if __name__ == "__main__":
    main()
