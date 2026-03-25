import time
from ec_func import readGraph, influence_spread_ic
from CovGreedy import select_seed_covgreedy
from CELF import celf_greedy
import random
import networkx as nx

n_subarea = 100
n_users = 3000
n_seed = 5
nodefilename = 'data/input_node_3000_0.txt'
edgefilename = 'data/input_edge_3000_0.txt'

def random_select_seed(G, seed_num):
    return set(random.sample(list(G.nodes), seed_num))

def select_seed_deggreedy(G, seed_num):
    node_degrees = [(node, len(list(G.neighbors(node)))) for node in G.nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    seeds = set([node for node, deg in node_degrees[:seed_num]])
    return seeds

def eval_algo(name, select_func, G, n_seed, n_subarea, num_simulations=100):
    t0 = time.time()
    seeds = select_func(G, n_seed)
    t1 = time.time()
    ec = influence_spread_ic(G, seeds, n_subarea, num_simulations)
    print(f"{name}: EC={ec:.4f}, time={t1-t0:.2f}s, seeds={sorted(seeds)}")
    return ec

def main():
    G = readGraph(nodefilename, edgefilename, n_subarea)
    print("===== Baseline Comparison =====")
    eval_algo("Random", random_select_seed, G, n_seed, n_subarea)
    eval_algo("DegGreedy", select_seed_deggreedy, G, n_seed, n_subarea)
    eval_algo("CovGreedy", lambda G, n_seed: select_seed_covgreedy(G, n_seed, n_subarea), G, n_seed, n_subarea)
    eval_algo("CELF", lambda G, n_seed: celf_greedy(G, n_seed, n_subarea, num_simulations=100), G, n_seed, n_subarea)

if __name__ == '__main__':
    main()
