
__author__ = 'Sam Sze'
from queue import PriorityQueue as PQ # priority queue\
from copy import deepcopy

def ConSubgraph(g, S, V_, p):
    g_ = deepcopy(g)
    for v in S:
        for v_ in V_:
            if (v in g_[v_]):
                del g_[v_][v]
            if (v_ in g_[v]):
                del g_[v][v_]
    return g_
