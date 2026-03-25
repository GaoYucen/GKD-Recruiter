import re
import time
import networkx as nx
import numpy as np
from EffectiveCoverage_1 import read_from_txt, effective_coverage,probeffectivecoverage
from effectivecoverage import effective_cov
from graphreader import read_graph

n_subarea = 100
n_users = 3000
n_seed_list = [1000]
n_samplefile = 10

input_file_path = 'data'
output_result_file_prefix = 'log\DegGreedy'


#read info from txt file into a graph
def readGraph(G,nodefilename,edgefilename, n_subarea):
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

# K is the size of seed set, C is candidate set, h is the number of sensing area
def select_seed_deggreedy(G, seed_num):
    S = set()
    DegDict = {}    #key 值为deg value具有该deg的candidate
    Deglist = list()
    for cc in set(G._node):
        deg = set(nx.neighbors(G, cc)).__len__()
        if deg in DegDict.keys():
            DegDict[deg].append(cc)
        else:
            DegDict[deg] = list()
            DegDict[deg].append(cc)
            Deglist.append(deg)
    Deglist.sort(reverse=True)
    DegListIndex = 0
    while S.__len__() < seed_num:
        if DegDict[Deglist[DegListIndex]].__len__() != 0:
            cc = DegDict[Deglist[DegListIndex]].pop()
            S.add(cc)
        else:
            DegListIndex = DegListIndex+1
    return S


def resultDegGreedy(n_subarea, n_users, n_samplefile,
                    input_file_path, output_result_file):

    n = len(n_seed_list)
    writefile_EC = open(output_result_file_prefix+'_EC.txt', 'w')
    writefile_stdEC = open(output_result_file_prefix+'_stdEC.txt', 'w')
    writefile_runtime = open(output_result_file_prefix+'_runtime.txt', 'w')
    
    first_line = 'seed_num'
    for k in n_seed_list:
        first_line += ('\t'+str(k))
    first_line = first_line+'\n'
    writefile_EC.write(first_line)
    writefile_stdEC.write(first_line)
    writefile_runtime.write(first_line)
    g = dict()
    g_quality = dict()
    sample_file_list = [0]
    for i in sample_file_list:
        
        nodefilename = input_file_path+'/input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path+'/input_edge_' + str(n_users) + '_' + str(i) + '.txt'
        
        ECresult_line = 'Input '+str(i)
        stdECresult_line = 'Input '+str(i)
        runtimeresult_line = 'Input '+str(i)

        G = nx.Graph()
        g.clear()
        g_quality.clear()
        g, g_quality = read_from_txt(nodefilename, edgefilename, n_subarea)
        G = readGraph(G, nodefilename, edgefilename, n_subarea)
        for j in range(0, n):
            n_seed = n_seed_list[j]
            start = time.process_time()
            S = select_seed_deggreedy(G, n_seed)
            end = time.process_time()
            t = end - start
            nx_G = read_graph(nodefilename, edgefilename, n_subarea)
            EC, stdEC = effective_cov(nx_G, S, n_subarea)
            ECresult_line += ('\t'+str(format(EC, '.4f')))
            stdECresult_line += ('\t'+str(format(stdEC, '.4f')))
            runtimeresult_line += ('\t'+str(format(t,'.4f')))
            print('Sample: ',i,' Seed number: ', n_seed, ' EC: ', format(EC, '.4f'), ' stdEC: ', format(stdEC, '.4f'), ' t: ', format(t,'.4f'))
        
        ECresult_line += '\n'
        stdECresult_line += '\n'
        runtimeresult_line += '\n'
        writefile_EC.write(ECresult_line)
        writefile_stdEC.write(stdECresult_line)
        writefile_runtime.write(runtimeresult_line)
        
        del G

    writefile_EC.close()
    writefile_stdEC.close()
    writefile_runtime.close()
    
# resultDegGreedy(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)