import networkx as nx


def read_graph(node_file_name, edge_file_name, n_subarea):

    G = nx.DiGraph()

    nodefile = open(node_file_name)
    newnode = nodefile.readline()
    while newnode:
        nodeId = int(newnode.split('\t')[0])
        nodeWeight = list()
        for i in range(0, n_subarea):
            nodeWeight.append(float(newnode.split('\t')[i+1]))
        G.add_node(nodeId, weight=nodeWeight)
        newnode = nodefile.readline()
    nodefile.close()

    edgefile = open(edge_file_name)
    newedge = edgefile.readline()
    while newedge:
        node1 = int(newedge.split('\t')[0])
        node2 = int(newedge.split('\t')[1])
        edgeWeight = float(newedge.split('\t')[2])
        G.add_weighted_edges_from([(node1, node2, edgeWeight)])
        newedge = edgefile.readline()
    edgefile.close()

    return G