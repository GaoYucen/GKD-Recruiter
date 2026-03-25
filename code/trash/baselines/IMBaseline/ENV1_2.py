import time
import math

from random import random
from copy import deepcopy

import networkx as nx
import numpy as np

# from degreeDiscount import degreeDiscountIC2
# from randomHeuristic import randomHeuristic

# number of neigh

class Env:
    ''' The General ENV, namely the graph
    Input: file -- address of the graph data
    Input: budget -- size of budget
    Comment: the structure in which we run our model
    '''
    def __init__(
    self,
    file,
    budget = 10
    ):
        self.netInput = []
        self.graph ={}
        self.seed = set()

        self.budget = budget
        self.graph = read_from_txt(file)
        self.graph_ = deepcopy(self.graph)
        self.file = file
        # Virtual graph
        # Node first tested on it
        # If accepted, replace graph with graph_

        self.list = strength_list(self.graph, file, self.budget)
        self.maxGain = self.list[0][1]

    def update_graph(self):
        '''
        Replace graph by graph_
        '''
        self.graph = self.graph_
        print("graph updated")

    def backup_graph(self):
        '''
        Update the graph_
        '''
        self.graph_ = deepcopy(self.graph)

    def list(self):
        '''
        Output: list of nodes ordered by degree with marginal gain of each
        '''
        return self.list

    def steps(self, step, A, T):
        '''
        Make action, steps to next stage
        Input: step -- index of node stepped
        Input: A -- bool, if this node is really added into seed set
        Input: T -- bool, if we are trainning
        Output: R -- Reward, marginal gain of a node, later used as machine Reward
        '''
        node = {self.list[step][0]}

        R = IC(self.graph, self.seed | node) - IC(self.graph, self.seed)

        # If this action is really made
        # 1. Add the node into seed set
        # 2. Update the graph (Collided Nodes)
        if A == 1:
            self.seed = self.seed | node
            self.update_graph()

        return R

    def node2feat(self,step):
        '''
        Covert a node index to its attributes: individual influence and collision
        backup graph in this stage, since it's the first step to work
        Input: step -- index of current node in self.list
        Output (private): self.netInput
        1. influence ratio
        2. collision ratio (self not included)
        Output: influence -- influence of individual node
        Output: coll -- collision of the node with seed set
        '''
        # Preparation
        # Update the graph_
        node = {self.list[step][0]}
        self.backup_graph()

        influence, coll, inf, touch = collision(self.graph, self.graph_, node, 0)
        if influence > 0.0:
            self.netInput = [touch / (influence), inf / (influence), coll / (influence)]
        else:
            self.netInput = [0, 0, 0]
        return influence, coll

    def greedy(self, budget):
        g = self.graph
        seed_g = set()
        for i in range(budget):
            node = {self.list[i][0]}
            seed_g = seed_g | node

        print(budget, IC(g, seed_g))

def read_from_txt(filename, p=0.01):
    """
        From TXT file read node pairs and assign active probability to edge by random
        Input: filename -- TXT file address
        Input: p -- weight of each edge
        Output: g, a graph in dict
    """
    print("started reading text")
    g = {}
    with open(filename) as f:
        # lines = f.readlines()[4:]
        lines = f.readlines()
        for line in lines:
            line = line.replace('\t', ' ')
            s = line.replace('\n', '').split(' ')
            e = list()
            e.append(int(s[0]))
            e.append(int(s[1]))
            e.append(float(s[2]))
            # print(e)
            # e = [int(s) for s in line.replace('\n', '').split(' ')]
            # e = [int(s) for s in line.replace('\n', '').split(' ')]
            r = 1 - e[2]
            if e[0] in g.keys():
                if e[1] in g[e[0]].keys():
                    x = g[e[0]][e[1]]
                    g[e[0]][e[1]] = 1 + (x - 1) * r
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

            # if g[e[0]][e[1]] != e[2] and g[e[1]][e[0]] != e[2]:
            #     print(g[e[0]][e[1]], g[e[1]][e[0]])

    print("finised reading text")
    print(g[e[0]][e[1]])
    return g

def collision(graph,graph_,Node, R):
    """
    calculate a new node's collison with seed set
    :param g: the graph
    :param A: new seed
    :param R: if this collison should be recorded
    :return: influence
    """

    res = 0.0
    coll = 0.0
    inf = 0.0
    touch = 0.0
    IC_ITERATION = 10

    # if R ==0, record everything on graph_, the backup graph
    if R == 0:
        g = graph_
    else:
        g = graph

    # recorded influence and collision of individual node
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in Node:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            g[w]["On"] += -1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
            res += len(total_activated_nodes)

    # record "On" nodes, the potentially activated nodes
    for _ in range(10):
        total_activated_nodes = set()
        for u in Node:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            if g[w]["On"] <= -IC_ITERATION * 1.2:
                                coll+= 1
                            if g[w]["On"] <= -IC_ITERATION * 0.5 and g[w]["On"] >= -IC_ITERATION:
                                inf += 1
                            if g[w]["On"] >= -IC_ITERATION * 0.5 and g[w]["On"] <= 0:
                                touch += 1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)

    end = time.time()
    #print("time cost on coll: ",end - start)
    return res / IC_ITERATION, (coll) /10, inf/10, touch/10 #to include itself

def IC(g, A):
    """
    calculate seed set's influence in IC（Independent Cascade）model
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    # print("IC running")
    res = 0.0
    IC_ITERATION = 100 # 300
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in A:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    #print(g[v].items(),flush=True)
                    for w, weight in g[v].items():
                        #print(w,weight,flush=True)
                        r = random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
        res += len(total_activated_nodes)
        # print("res:", res)
        # print("IC ended")
    return res / IC_ITERATION

def degreeDiscount(G, k, g):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (without priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    Note: the routine runs twice slower than using PQ. Implemented to verify results
    '''
    d = dict()
    dd = dict()  # degree discount
    t = dict()  # number of selected neighbors
    S = []  # selected set of nodes
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(iter(dd.items()), key=lambda k_v: k_v[1])
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']  # increase number of selected neighbors
                # print(u, v)
                try:
                    dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * g[u][v]
                    # print(g[u][v])
                except:
                    dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * g[v][u]
    return S

def strength_list(g, file, budget):
    '''
    Input: budget -- size of seed set
    Return: A -- budget*20 nodes, ordered by degree and individual influence attached
    '''
    print("started preparation")
    G = nx.Graph()
    # ----------modified----------
    with open(file) as f:
        f.readline()
        for line in f:
            u, v = list(map(int, line.split()[:2]))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)

    # S =degreeDiscount(G, budget*40, g)
    S = degreeDiscount(G, budget*1000, g)
    #T =randomHeuristic(G, budget*20, p=.01)
    print("1")
    # print(IC(g,S[0:budget]))
    #print("2")
    #print(IC(g,T[0:budget]))

    max_increment = {i:IC(g, {i}) for i in S}
    print("finished IC calculation")
    A = sorted(max_increment.items(), key=lambda x:x[1], reverse=True)
    print("finished sorting")
    # A = t[0:budget*20]
    # print(IC(g, A[0:budget]))
    print("finished preparation")
    return A;


