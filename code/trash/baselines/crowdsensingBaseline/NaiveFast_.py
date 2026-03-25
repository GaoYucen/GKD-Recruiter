import os
import re
import time
import numpy as np
import networkx as nx
from random import *
from EffectiveCoverage_1 import read_from_txt, effective_coverage, probeffectivecoverage
from effectivecoverage import effective_cov
from CovGreedy import select_seed_covgreedy
# from KTVoting import KTVoting2
from DegGreedy import select_seed_deggreedy

n_subarea = 100
n_users = 5000

# n_seeds_list = [500, 700, 900, 1000]
n_seeds_list = [100]
n_seed_list = []
for i in n_seeds_list:
    n_seed_list.append(int(1 / n_subarea))
budget = 1

n_samplefile = 10
iteration_ic = 100
input_path = os.path.join("data", "input_5000_0")
input_file_path = os.path.join("data", "New_Gowalla")
model_name = "_crowdsensing_baselines"
logpath = os.path.join("log", "input_5000_0", "random")

task_list = list()

for i in range(n_subarea):
    task_name = "task" + str(i)
    task_list.append(task_name)

data_list = task_list

baselines = ["NaiveFast", "DegGreedy", "CovGreedy"]
inf_dict = dict()
time_dict = dict()
task_dict = dict()
for i in baselines:
    inf_dict[i] = dict()
    time_dict[i] = dict()
    task_dict[i] = np.zeros([2, n_subarea])
    for j in n_seeds_list:
        inf_dict[i][j] = 0
        time_dict[i][j] = 0

def readGraph(G, nodefilename, edgefilename, n_subarea):
    nodefile = open(nodefilename)
    newnode = nodefile.readline()
    while newnode:
        nodeId = int(newnode.split('\t')[0])
        nodeWeight = list()
        for i in range(0, n_subarea):
            nodeWeight.append(float(newnode.split('\t')[i+1]))
        G.add_node(nodeId, weight=nodeWeight)
        newnode = nodefile.readline()
    edgefile = open(edgefilename)
    newedge = edgefile.readline()
    while newedge:
        node1 = int(newedge.split('\t')[0])
        node2 = int(newedge.split('\t')[1])
        edgeWeight =  float(newedge.split('\t')[2])
        G.add_weighted_edges_from([(node1, node2, edgeWeight)])
        G.add_weighted_edges_from([(node2, node1, edgeWeight)])
        newedge = edgefile.readline()
    return G


def cos_sim(x,y):
    a = np.mat(x)
    b = np.mat(y)
    num = float(a * b.T)
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    if denom == 0:
        return 0
    else:
        sim = 0.5 + 0.5 * (num / denom)
        return sim


def FastSelector(k, beta, h, G):
    # k is the number of seeds
    # beta is the paramter of DegreeRank and TriDiffRank
    # h is the number of subareas
    S = set()
    DegreeAll = dict(G.degree())
    apVector = np.zeros(h)
    while len(S)<k:
        # Calclate arg max R(u)
        TriCosSim = dict()
        Degree = dict()
        DegreeRank = dict()
        TriDiffRank = dict()
        for v in set(G._node):
            if v not in S:
                ap = np.array(G._node[v]['weight'])
                TriCosSim[v] = cos_sim(ap,apVector)
                Degree[v] = DegreeAll[v]
        DegreeRankInd = sorted(Degree.items(), key=lambda x: x[1], reverse=True)
        TriDiffRankInd = sorted(TriCosSim.items(), key=lambda x: x[1], reverse=True)
        for i in range(0,len(DegreeRankInd)):
            DegreeRank[DegreeRankInd[i][0]] = i+1
            TriDiffRank[TriDiffRankInd[i][0]] = i+1
        minR = 2*len(DegreeRank)
        u = 0
        for v in DegreeRank.keys():
            R = beta*DegreeRank[v]+(1-beta)*TriDiffRank[v]
            if R<minR:
                minR = R
                u = v
        S.add(u)
    return S

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
                    try:
                        for w, weight in g[v].items():
                            r = random()
                            if w not in total_activated_nodes and r < weight:
                                l2.add(w)
                                total_activated_nodes.update({w})
                    except:
                        pass
                l1 = l2
                l2 = set()
        res += len(total_activated_nodes)
    return res / IC_ITERATION


def resultFastSelector(budget, n_subarea, n_users, input_file_path, file):

    # n = len(n_seed_list)
    
    g = dict()
    g_quality = dict()
    samplefilelist = [0]
    for i in samplefilelist: #range(0, n_samplefile):
        G = nx.Graph()
        nodefilename = input_file_path+'/input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path+'/input_edge_' + str(n_users) + '_' + str(i) + '.txt'
        
        g.clear()
        g_quality.clear()
        
        g, g_quality = read_from_txt(file)
        G = readGraph(G, nodefilename, edgefilename, n_subarea)
        for j in range(1):
            # n_seed = n_seed_list[j]
            start = time.process_time()
            S = FastSelector(budget, 0.56, n_subarea, G)
            end = time.process_time()
            t = end - start
            print(S)
            inf = IC(g, S)
            print("NaiveFast")
            print(budget, ' ', inf , ' ', t)
            print("NaiveFast ", budget, ' ', inf , ' ', t, ' ', file=log, flush=True)
            print(S)
            print(S, file=log, flush=True)
        return inf, t


def resultDegGreedy(budget, n_subarea, n_users, input_file_path, file):
    # n = len(n_seed_list)

    g = dict()
    g_quality = dict()
    samplefilelist = [0]
    for i in samplefilelist:  # range(0, n_samplefile):
        G = nx.Graph()
        nodefilename = input_file_path + '\input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path + '\input_edge_' + str(n_users) + '_' + str(i) + '.txt'

        g.clear()
        g_quality.clear()

        g, g_quality = read_from_txt(file)
        G = readGraph(G, nodefilename, edgefilename, n_subarea)
        for j in range(1):
            # n_seed = n_seed_list[j]
            start = time.process_time()
            S = select_seed_deggreedy(G, budget)
            end = time.process_time()
            t = end - start
            inf = IC(g, S)
            print("DegGreedy")
            print(budget, ' ', inf, ' ', t)
            print("DegGreedy ", budget, ' ', inf, ' ', t, ' ', file=log, flush=True)
            print(S)
            print(S, file=log, flush=True)
        return inf, t

def resultCovGreedy(budget, n_subarea, n_users, input_file_path, file):
    # n = len(n_seed_list)

    g = dict()
    g_quality = dict()
    samplefilelist = [0]
    for i in samplefilelist:  # range(0, n_samplefile):
        G = nx.Graph()
        nodefilename = input_file_path + '\input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path + '\input_edge_' + str(n_users) + '_' + str(i) + '.txt'

        g.clear()
        g_quality.clear()

        g, g_quality = read_from_txt(file)
        G = readGraph(G, nodefilename, edgefilename, n_subarea)
        for j in range(1):
            # n_seed = n_seed_list[j]
            start = time.process_time()
            S = select_seed_covgreedy(G, budget, n_subarea)
            end = time.process_time()
            t = end - start
            inf = IC(g, S)
            print(budget, ' ', inf, ' ', t)
            print("CovGreedy ", budget, ' ', inf, ' ', t, ' ', file=log, flush=True)
            print(S)
            print(S, file=log, flush=True)
        return inf, t

def resultKTVoting(budget, n_subarea, n_users, input_file_path, file):
    # n = len(n_seed_list)

    g = dict()
    g_quality = dict()
    samplefilelist = [0]
    for i in samplefilelist:  # range(0, n_samplefile):
        G = nx.Graph()
        nodefilename = input_file_path + '\input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path + '\input_edge_' + str(n_users) + '_' + str(i) + '.txt'

        g.clear()
        g_quality.clear()

        g, g_quality = read_from_txt(file)
        G = readGraph(G, nodefilename, edgefilename, n_subarea)
        for j in range(1):
            # n_seed = n_seed_list[j]
            start = time.process_time()
            S = KTVoting2(budget,n_subarea, 0.9, 4, G)
            end = time.process_time()
            t = end - start
            inf = IC(g, S)
            print(budget, ' ', inf, ' ', t)
            print("KTVoting ", budget, ' ', inf, ' ', t, ' ', file=log, flush=True)
            print(S)
            print(S, file=log, flush=True)
        return inf, t

num_seeds = n_seeds_list[0]
log_path = logpath + str(num_seeds) + model_name + "_1.txt"
# log_path = log_path + data + ".txt"
print(log_path)
log = open(log_path, "w")

# for data in data_list:
    # count += 1

for n in n_seeds_list:
    for _ in range(n):
        k = randint(0, n_subarea-1)
        data = data_list[k]
        file = input_path + data + ".txt"

        # log_path = logpath + data + ".txt"

        print("NaiveFast")
        inf1, t1 = resultFastSelector(budget, n_subarea, n_users, input_file_path, file)
        inf_dict["NaiveFast"][num_seeds] += inf1
        time_dict["NaiveFast"][num_seeds] += t1
        task_dict["NaiveFast"][0][k] += 1
        task_dict["NaiveFast"][1][k] += inf1

        print("DegGreedy")
        inf2, t2 = resultDegGreedy(budget, n_subarea, n_users, input_file_path, file)
        inf_dict["DegGreedy"][num_seeds] += inf2
        time_dict["DegGreedy"][num_seeds] += t2
        task_dict["DegGreedy"][0][k] += 1
        task_dict["DegGreedy"][1][k] += inf2

        print("CovGreedy")
        inf3, t3 = resultCovGreedy(budget, n_subarea, n_users, input_file_path, file)
        inf_dict["CovGreedy"][num_seeds] += inf3
        time_dict["CovGreedy"][num_seeds] += t3
        task_dict["CovGreedy"][0][k] += 1
        task_dict["CovGreedy"][1][k] += inf3

        # inf4, t4 = resultKTVoting(budget, n_subarea, n_users, input_file_path, file)
        # inf_dict["KTVoting"][num_seeds] += inf4
        # time_dict["KTVoting"][num_seeds] += t4

log.close()

for i in n_seeds_list:
    log_tot = logpath + str(i) + model_name + "_tot_1.txt"
    log1 = open(log_tot, "w")
    for j in baselines:
        print(j, ' ', str(i), ' ', inf_dict[j][i], ' ', time_dict[j][i], ' ', file=log1, flush=True)
        print(*task_dict[j][0], file=log1, flush=True)
        print(*task_dict[j][1], file=log1, flush=True)
    log1.close()


# resultFastSelector(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)