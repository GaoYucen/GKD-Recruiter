from random import random

def LazyForward(g,k):
    V = set(g.keys())
    max_increment = {i:IC(g, {i}) for i in g.keys()}
    t_max_increment = max_increment.copy()
    t = sorted(max_increment.items(), key=lambda x:x[1], reverse=True)
    A = set([t[0][0]])
    max_influence = t[0][1]
    del t_max_increment[t[0][0]]
    for _ in range(k - 1):
        max_increment_current = 0
        max_increment_node = t[0][0]
        t = sorted(t_max_increment.items(), key=lambda x:x[1], reverse=True)
        for v, _ in t:
            if v in t_max_increment:
                if max_increment[v] > max_increment_current:
                    increment_A_and_v = IC(g, A | {v}) - max_influence
                    if max_increment_current < increment_A_and_v:
                        max_increment_current = increment_A_and_v
                        max_increment[v] = t_max_increment[v] = increment_A_and_v
                        max_increment_node = v
        A.add(max_increment_node)
        max_influence = max_influence + max_increment_current
        del t_max_increment[max_increment_node]
        print(max_increment_node)
    return A, max_influence

def IC(g, A, IC_ITERATION = 10):
    """
    calculate seed set's influence in IC（Independent Cascade）model
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    res = 0.0
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in A:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            total_activated_nodes.update({u})
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random()
                        if w not in total_activated_nodes and r < weight:
                            l2.add(w)
                            total_activated_nodes.update({w})
                l1 = l2
                l2 = set()
        res += len(total_activated_nodes)
    return res / IC_ITERATION
