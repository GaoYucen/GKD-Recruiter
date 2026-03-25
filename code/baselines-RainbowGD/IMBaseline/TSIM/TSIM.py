from TSIM.ConSubgrpah import *
from TSIM.degreeDiscount import *
from TSIM.LazyForward import *

def IcRange(g, A, IC_ITERATION = 10):
    """
    calculate seed set's influence in IC（Independent Cascade）model
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    res = 0.0
    total_nodes_history = set()
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
                        r =  random()
                        if w not in total_activated_nodes and r < weight:
                            l2.add(w)
                            total_activated_nodes.update({w})
                            total_nodes_history.update({w})
                l1 = l2
                l2 = set()
        res += len(total_activated_nodes)
    return res / IC_ITERATION,total_nodes_history

def TSIM(g, G, k, p):
    S = degreeDiscountIC2(G, k, p)
    _, V_ = IcRange(g,S)
    G_ = ConSubgraph(g, S, V_, p)
    S_, _ = LazyForward(G_, k)
    temp = dict()
    result = []
    S = set(S)
    S_ = set(S_)
    Snew = S.union(S_)
    for u in Snew:
        deg = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        inf = IC(g,{u})
        MIV = p*deg + (1-p)*inf
        temp[u] = MIV
    for i in range(k):
        u, ddv = max(iter(temp.items()), key=lambda k_v: k_v[1])
        temp.pop(u)
        result.append(u)
    return result
