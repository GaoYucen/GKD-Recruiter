import time
import math
from random import *
from copy import deepcopy
import networkx as nx
from TSIM.TSIM import *
import os
import psutil
import numpy as np

budget_tot = 100 # k
iteration_ic = 100 # number of influence simulations
task_num = 100
budget = 1

# filename = list()
filepath = "E:\Research\paper\\2022crowdsensing\\newdata\\input_5000_0\\"
# filepath = "E:\Research\paper\\2022crowdsensing\\data\\New_Gowalla\\input_edge_3000_0.txt"
# logpath = "E:\Research\paper\\2022crowdsensing\log\\input_3000_0\\"
logpath = "E:\Research\paper\\2022crowdsensing\\log\\input_5000_0\\random\\"
# baselines = ["Degree", "Random", "CELF", "TSIM"]
model_name = "_IM_baselines"
baselines = ["Degree", "CELF", "TSIM"]
inf_dict = dict()
time_dict = dict()
S_dict = dict()
task_dict = dict()
for i in baselines:
    inf_dict[i] = 0
    time_dict[i] = 0
    S_dict[i] = set()
    task_dict[i] = np.zeros([2, task_num])

task_list = list()
for i in range(task_num):
    task_name = "task" + str(i)
    task_list.append(task_name)

data_list = task_list

inf_Random = 0
inf_Degree = 0
inf_CELF = 0
inf_TSIM = 0

# file = r"E:\Research\paper\2021-lym\Final Code\Final Code\Experiment\data\transCit-HepPh-test.txt"
# log = open(r"E:\Research\paper\2021-lym\Final Code\Final Code\Experiment\log_1\test1.txt","w")

process = psutil.Process(os.getpid())

class Env:
    ''' The General ENV, namely the graph
    Input: file -- address of the graph data
    Input: budget -- size of budget
    Comment: the structure in which we run our model
    '''
    def __init__(self, file, budget = 50):
        start_M = process.memory_info().rss  # in bytes
        self.netInput = []
        self.G = []
        self.graph ={}
        self.quality = {}
        self.seed = set()
        self.budget = budget
        self.graph, self.quality = read_from_txt(file)
        self.graph_ = deepcopy(self.graph)
        end_M = process.memory_info().rss  # in bytes
        print("Memory used",end_M - start_M, "bytes", flush=True)  # in bytes
        # Virtual graph
        # Node first tested on it
        # If accepted, replace graph with graph_
        print("Clock")
        start = time.time()
        self.list = []
        self.maxGain = []
        print(time.time()-start)
        print("Cock")
        end_M = process.memory_info().rss  # in bytes
        print("Memory used",end_M - start_M, "bytes", flush=True)  # in bytes

        #print(self.graph)
    def update_graph(self):
        '''
        Replace graph by graph_
        '''
        temp = self.graph
        self.graph = self.graph_
        del(temp)

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

def IC(g, A, IC_ITERATION = iteration_ic):
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

def randomHeuristic(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    import random
    S = random.sample(G.nodes(), k)
    return S

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
    dd = dict() # degree discount
    t = dict() # number of selected neighbors
    S = [] # selected set of nodes
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(iter(dd.items()), key=lambda k_v: k_v[1])
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                # print(u, v)
                try:
                    dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*g[u][v]
                    # print(g[u][v])
                except:
                    dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * g[v][u]
                    # print(g[v][u])
    return S

def CELF(g, k, log, process, n):
    """
    CELF算法计算最大影响力的k个节点
    :param g: 图
    :param k: 节点数量
    :return: (最大影响力节点集合（以set存储）, 最大影响力)
    """
    start = time.time()
    start_M = process.memory_info().rss
    V = set(g.keys())
    max_increment = {i:IC(g, {i}) for i in g.keys()}
    t_max_increment = max_increment.copy()
    t = sorted(max_increment.items(), key=lambda x:x[1], reverse=True)
    A = set([t[0][0]])
    max_influence = t[0][1]
    del t_max_increment[t[0][0]]
    for _ in range(k):
        # if len(A)%10 == 0:
        if len(A) == k:
            temp = time.time()
            inf3 = IC(g, A)
            print(len(A),' ',inf3,' ',temp - start,' ',file=log, flush=True)
            print(A)
            print(A, file=log, flush=True)
            inf_dict["CELF"] += inf3
            time_dict["CELF"] += temp - start
            S_dict["CELF"] = S_dict["CELF"] | set(A)
            task_dict["CELF"][0][n] += 1
            task_dict["CELF"][1][n] += inf3
            start = start + time.time()-temp
        max_increment_current = 0
        max_increment_node = 0
        t = sorted(t_max_increment.items(), key=lambda x:x[1], reverse=True)
        for v, _ in t:
            if max_increment[v] > max_increment_current:
                increment_A_and_v = IC(g, A | {v}) - max_influence
                if max_increment_current < increment_A_and_v:
                    max_increment_current = increment_A_and_v
                    max_increment[v] = t_max_increment[v] = increment_A_and_v
                    max_increment_node = v
        A.add(max_increment_node)
        max_influence = max_influence + max_increment_current
        del t_max_increment[max_increment_node]
    end_M = process.memory_info().rss
    print("memory", end_M - start_M, file=log, flush=True)
    return A, max_influence



def OtherTime(file, g, budget, k):
    print(file,file = log)
    G = nx.Graph()
    maxInDegree = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            s = line.split()
            u, v = list(map(int, s[:2]))
            p = float(s[2])
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
            if G[u][v]['weight'] > maxInDegree:
                maxInDegree = G[u][v]['weight']

    print("Degree")
    print("Degree",file = log, flush=True)
    start_M = process.memory_info().rss
    # for i in range(10,budget,10):
    start = time.time()
    S = degreeDiscount(G, budget, g)
    Time = time.time() - start
    end_M = process.memory_info().rss
    inf1 = IC(g, S)
    print(budget, ' ', inf1, ' ', Time, ' ', file=log, flush=True)
    print(S)
    print(S, file=log, flush=True)
    print("memory", end_M - start_M, file = log,flush=True)
    inf_dict["Degree"] += inf1
    time_dict["Degree"] += Time
    S_dict["Degree"] = S_dict["Degree"] | set(S)
    task_dict["Degree"][0][k] += 1
    task_dict["Degree"][1][k] += inf1

    # print("Random")
    # print("Random",file = log, flush=True)
    # start_M = process.memory_info().rss
    # # for i in range(10,budget,10):
    # start = time.time()
    # S = randomHeuristic(G, budget)
    # Time = time.time() - start
    # end_M = process.memory_info().rss
    # inf2 = IC(g, S)
    # print(budget, ' ', inf2, ' ', Time, ' ', file=log, flush=True)
    # print(S)
    # print(S, file=log, flush=True)
    # # end_M = process.memory_info().rss
    # print("memory", end_M-start_M, file = log,flush=True)
    # inf_dict["Random"] += inf2
    # time_dict["Random"] += Time

    print("CELF")
    print("CELF", file=log, flush=True)
    # start_M = process.memory_info().rss
    CELF(g, budget, log, process, k)

    print("TSIM")
    print("TSIM", file=log, flush=True)
    start_M = process.memory_info().rss
    # for i in range(10, budget, 10):
    start = time.time()
    S = TSIM(g, G, budget, p)
    Time = time.time() - start
    end_M = process.memory_info().rss
    inf4 = IC(g, S)
    print(budget, ' ', inf4 , ' ', Time, ' ', file=log, flush=True)
    print(S)
    print(S, file=log, flush=True)
    # print(Time)
    print("memory", end_M - start_M, file=log, flush=True)
    inf_dict["TSIM"] += inf4
    time_dict["TSIM"] += Time
    S_dict["TSIM"] = S_dict["TSIM"] | set(S)
    task_dict["TSIM"][0][k] += 1
    task_dict["TSIM"][1][k] += inf4

# test = Env(file,10)
# OtherTime(file,test.graph,budget)
#print(IC(test.graph,S))
#print(test.list)
#print(test.graph)
#print(CELF.CELF_plus_plus(test.graph, 300))
#test = Env(r"C:\Users\siwei\OneDrive\CNA\大创\complex_network_course-master\Homework_4\DBLP.txt")
#print(CELF.CELF_plus_plus(test.graph, 300))

# count = 0 # only for testing

log_path = logpath + str(budget_tot) + model_name + "_1.txt"
print(log_path)
log = open(log_path, "w")
for _ in range(budget_tot):
    k = randint(0, len(data_list)-1)
    data = data_list[k]
    print(data)
    # count += 1
    file = filepath + data + ".txt"

    log_path = logpath + data + ".txt"
    log_path = logpath + str(budget_tot) + "\\"
    log_path = log_path + data + ".txt"
    print(log_path)

    test = Env(file, 10)
    OtherTime(file, test.graph, budget, k)

# file = filepath
# test = Env(file, 10)
# OtherTime(file, test.graph, budget_tot)
# log.close()

log_tot = logpath + str(budget_tot) + model_name + "_tot_1.txt"
log1 = open(log_tot, "w")
for i in baselines:
    print(i, ' ', budget_tot, ' ', inf_dict[i], ' ', time_dict[i], ' ', file=log1, flush=True)
    # print(S_dict[i], file=log1, flush=True)
    print(*task_dict[i][0], file=log1, flush=True)
    print(*task_dict[i][1], file=log1, flush=True)
    # print(i, ' ', budget_tot, ' ', IC(test.graph, S_dict[i]), ' ', time_dict[i], ' ', file=log1, flush=True)
log1.close()

