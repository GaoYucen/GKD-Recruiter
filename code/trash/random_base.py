import random
from ec_func import readGraph, influence_spread_ic

n_subarea = 100
n_users = 3000
n_seed = 5
nodefilename = 'data/Gowalla/input_node_3000_0.txt'
edgefilename = 'data/Gowalla/input_edge_3000_0.txt'

def random_select_seed(G, seed_num):
    """随机选择种子节点"""
    return set(random.sample(list(G.nodes), seed_num))

def main():
    G = readGraph(nodefilename, edgefilename, n_subarea)
    num_trials = 10
    ec_values = []
    for _ in range(num_trials):
        seeds = random_select_seed(G, n_seed)
        ec = influence_spread_ic(G, seeds, n_subarea, num_simulations=100)
        ec_values.append(ec)
        print(f"Trial ec value: {ec:.4f}")
    avg_ec = sum(ec_values) / num_trials
    print(f"\nAverage ec value over {num_trials} trials: {avg_ec:.4f}")

if __name__ == '__main__':
    main()
