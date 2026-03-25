import random
import time
import math
import numpy as np
from copy import deepcopy

MC_IC_ITERATION = 300

# def read_from_txt(edge_file_path):
#     """
#     From TXT file read node pairs and assign active probability to edge by random
#     :param: filename, TXT file name
#     :return: g, a graph in dict
#     """
#     g_quality = {}
#     g = {}
#     with open(edge_file_path) as f:
#         lines = f.readlines()[1:]
#         for line in lines:
#             e0 = int(line.replace('\n', '').split(' ')[0])
#             e1 = int(line.replace('\n', '').split(' ')[1])
#             r = float(line.replace('\n', '').split(' ')[2])
#
#             if e0 in g.keys():
#                 g[e0]['edge'][e1] = r
#             else:
#                 g[e0] = {}
#                 g[e0]['hit'] = 0
#                 g[e0]['edge'] = {}
#                 g[e0]['edge'][e1] = r
#             #if e1 in g.keys():
#             #    g[e1]['edge'][e0] = r
#             #else:
#             #    g[e1] = {}
#             #    g[e1]['hit'] = 0
#             #    g[e1]['edge'] = {}
#             #    g[e1]['edge'][e0] = r
#
#     return g, g_quality

def read_from_txt(filename):
    """
    From TXT file read node pairs and assign active probability to edge by random
    Input: filename -- TXT file address
    Input: p -- weight of each edge
    Output: g, a graph in dict
    snap, hundreds of snapshots
    """
    g = {}
    q = {}

    with open(filename) as f:
        # lines = f.readlines()[4:]
        lines = f.readlines()
        for line in lines:
            line = line.replace('\t',' ')
            s = line.replace('\n', '').split(' ')
            e = list()
            e.append(int(s[0]))
            e.append(int(s[1]))
            e.append(float(s[2]))
            # e.append(float(s[3]))
            # print(e)
            # e = [int(s) for s in line.replace('\n', '').split(' ')]
            #e = [int(s) for s in line.replace('\n', '').split(' ')]
            r = 1 - e[2]
            if e[0] in g.keys():
                if e[1] in g[e[0]].keys():
                    x = g[e[0]][e[1]]
                    g[e[0]][e[1]] = 1 + (x-1)*r
                else:
                    g[e[0]][e[1]] = e[2]
            else:
                g[e[0]] = {"On": 0}
                g[e[0]][e[1]] = e[2]

            if e[1] in g.keys():
                pass
            else:
                g[e[1]] = {"On": 0}
                g[e[1]][e[0]] = 0

            # q[e[1]] = e[3]
            # if g[e[0]][e[1]] != e[2] and g[e[1]][e[0]] != e[2]:
                # print(g[e[0]][e[1]], g[e[1]][e[0]])
    return g, q


def effective_coverage(g, g_quality, A, n_subareas):
    """
    calculate a node's effective coverage in IC（Independent Cascade）model by Monte Carlo
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    hit_times = {}
    res = 0.0
    for node in g.keys():
        g[node]['hit'] = 0

    for _ in range(MC_IC_ITERATION):
        total_activated_nodes = set()
        activated_nodes = set()
        for u in A:
            current_influential_nodes = {u}   # current_influential_nodes
            next_influential_nodes = set()  # next_influential_nodes
            activated_nodes.add(u)
            while len(current_influential_nodes):
                for v in current_influential_nodes:
                    if g[v]['edge'] is not None:
                        for w, weight in g[v]['edge'].items():
                            r = random.random()
                            # each node should only be activated once
                            if r < weight and w not in activated_nodes:
                                g[w]['hit'] += 1
                                next_influential_nodes.add(w)
                                activated_nodes.add(w)
                current_influential_nodes = next_influential_nodes
                next_influential_nodes = set()
            total_activated_nodes.update(activated_nodes)

    # Calculate effective coverage according to its definition
    EC = np.zeros(n_subareas)
    for i in range(0,n_subareas):
        for node in g.keys():
            if node in g_quality.keys():
                EC[i] += g_quality[node][i]*g[node]['hit']/MC_IC_ITERATION

    return np.mean(EC), np.std(EC)


def probeffectivecoverage(g,g_quality, S, n_subareas):
    nodeAp = dict()
    for key in g.keys():
        if key in S:
            nodeAp[key] = float(0)
        else:
            nodeAp[key] = float(1)
    IAS = set()
    IAS = IAS.union(set(g.keys())) - S
    NIS = set()
    NIS = NIS.union(S)
    CIS = set()
    while not not IAS and not not NIS:
        CIS.clear()
        CIS = CIS.union(NIS)
        NIS.clear()
        for i in CIS:
            if i in set(g.keys()):
                for j in set(g[i]['edge'].keys()):
                    nodeAp[j] = (1 - g[i]['edge'][j] * (1 - nodeAp[i])) * nodeAp[j]
                    if j in IAS:
                        NIS.add(j)
        IAS = IAS - NIS
    for u in set(g.keys()):
        nodeAp[u] = 1 - nodeAp[u]

    # Calculate effective coverage according to its definition
    EC = np.zeros(n_subareas)
    for i in range(0,n_subareas):
        for node in g.keys():
            if node in g_quality.keys():
                EC[i] += nodeAp[node]*g_quality[node][i]

    return np.mean(EC), np.std(EC), EC

def effectivecoverage2(g,g_quality, S, n_subareas):
    nodeAp = dict()
    for key in g.keys():
        if key in S:
            nodeAp[key] = float(0)
        else:
            nodeAp[key] = float(1)
    IAS = set()
    IAS = IAS.union(set(g.keys())) - S
    NIS = set()
    NIS = NIS.union(S)
    CIS = set()
    while not not IAS and not not NIS:
        CIS.clear()
        CIS = CIS.union(NIS)
        NIS.clear()
        for i in CIS:
            if i in set(g.keys()):
                for j in set(g[i]['edge'].keys()):
                    nodeAp[j] = (1 - g[i]['edge'][j] * (1 - nodeAp[i])) * nodeAp[j]
                    if j in IAS:
                        NIS.add(j)
        IAS = IAS - NIS
    for u in set(g.keys()):
        nodeAp[u] = 1 - nodeAp[u]

    # Calculate effective coverage according to its definition
    EC = np.zeros(n_subareas)
    for i in range(0,n_subareas):
        for node in g.keys():
            if node in g_quality.keys():
                EC[i] += nodeAp[node]*g_quality[node][i]

    return np.mean(EC), np.std(EC)


def IC(g, g_quality, A, n_subareas):
    """
    calculate a node's influence in IC（Independent Cascade）model
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    res = 0.0
    IC_ITERATION = 300
    EC = np.zeros(n_subareas)
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in A:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v]['edge'].items():
                        r = random.random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
        for i in range(0,n_subareas):
            for node in total_activated_nodes:
                if node in g_quality.keys():
                    EC[i] += g_quality[node][i]
                    
    EC = EC/IC_ITERATION
    return np.mean(EC), np.std(EC)



