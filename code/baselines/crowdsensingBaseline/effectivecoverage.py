import numpy as np


def effective_cov(nx_G, seedset, n_subareas):
    """
    calculate a node's effective coverage in IC（Independent Cascade）model by Monte Carlo
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """

    paticipate_prob = dict()
    quality = dict()
    all_nodes = list(nx_G.nodes())
    for node in all_nodes:
        quality[node] = np.array(nx_G.nodes[node]['weight'])
        if node in seedset:
            paticipate_prob[node] = 1
        else:
            paticipate_prob[node] = 0

    all_activated_node = set(seedset)
    current_active_node = set(seedset)
    next_active_node = set()
    for node in current_active_node:
        for nbr in nx_G.neighbors(node):
            if nbr not in all_activated_node:
                next_active_node.add(nbr)

    while len(current_active_node) != 0 and len(next_active_node) != 0:
        for node in current_active_node:
            for nbr in nx_G.neighbors(node):
                if nbr in next_active_node:
                    paticipate_prob[nbr] = 1-(1-paticipate_prob[nbr])*(1-paticipate_prob[node]*nx_G[node][nbr]['weight'])
        current_active_node = next_active_node.copy()
        all_activated_node = all_activated_node | current_active_node
        next_active_node = set()
        for node in current_active_node:
            for nbr in nx_G.neighbors(node):
                if nbr not in all_activated_node:
                    next_active_node.add(nbr)

    EC = np.zeros(n_subareas)
    for node in paticipate_prob.keys():
        if node in quality.keys():
            EC += paticipate_prob[node] * quality[node]

    return np.mean(EC), np.std(EC)
