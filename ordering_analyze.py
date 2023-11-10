import data
import node_ordering_big
import numpy as np
import networkx as nx
from scipy import sparse
import random

def permute(G):
    node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
    # build a new graph
    G_new = nx.relabel_nodes(G, node_mapping)
    return G_new

def permutation_test(adj):
    permuted_adj_list = []
    for i in range(1):
        or_G = nx.from_scipy_sparse_matrix(adj)
        G= node_ordering_big.random_shuffle(or_G)
        Reordered_G = node_ordering_big.get_graph_data(G, "DFS", order_only=False)
        permuted_adj_list.append(nx.adjacency_matrix(Reordered_G[0]))
    return permuted_adj_list

def numeic_analyze(permuted_adj_list):
    # whats the expectation of having 1 in specific element of adj under any permutation
    sum = permuted_adj_list[0]

    for i in range(1, len(permuted_adj_list)):
        sum += permuted_adj_list[i]
    return np.array(sum.data)


# data_set_name = "DD"
# adj_list,_,_ = data.list_graph_loader(data_set_name)
import randomGraphGen

#------------------------------------

import torch_geometric
import numpy
import scipy
# dataset_b = torch_geometric.datasets.MNISTSuperpixels(root = "data/geometric")
# #
# list_adj = []
# list_x = []
# for i in range(len(dataset_b.data.y)):# len(dataset_b.data.y)
#     in_1 = dataset_b[i].edge_index[0].detach().numpy()
#     in_2 = dataset_b[i].edge_index[1].detach().numpy()
#     valu=  numpy.ones(len(in_2))
#     adj = scipy.sparse.csr_matrix((valu,(in_1, in_2)), shape=(dataset_b[0].num_nodes, dataset_b[0].num_nodes))
#     list_adj.append(adj)
#     list_x.append(dataset_b[i].y)
#------------------------------------

# graphGen = randomGraphGen.GraphGenerator()
# adj_list = [nx.adjacency_matrix(graphGen(100,"random_geometric")[0]) for i in range(20)]
# adj_list_ =  [nx.from_scipy_sparse_matrix(list_adj[0]) for g in list_adj]

#-------------------------------
# dataset_b = torch_geometric.datasets.ZINC(root = "data/MoleculeNet/zinc",subset=True)
# list_adj = []
# for i in range(len(dataset_b.data.y)):
#     in_1 = dataset_b[i].edge_index[0].detach().numpy()
#     in_2 = dataset_b[i].edge_index[1].detach().numpy()
#     valu=  numpy.ones(len(in_2))
#     adj = scipy.sparse.csr_matrix((valu,(in_1, in_2)), shape=(dataset_b[i].num_nodes, dataset_b[i].num_nodes))
#     list_adj.append(adj)
#     label = dataset_b[0].y
re= data.list_graph_loader("mnist")
list_adj = re[0]
list_adj = data.Datasets(list_adj, True, [None for x in list_adj],None, set_diag_of_isol_Zer=False).processed_adjs
#-------------------------------



adj_list = list_adj
permuted_graphs = []
iso_exp_1 = []
for graph in adj_list:
    all_iso_of_g_ = permutation_test(graph)
    iso_exp_1.append(numeic_analyze(all_iso_of_g_))
    permuted_graphs.extend(all_iso_of_g_)
hit_number = (numeic_analyze(permuted_graphs))


import matplotlib.pyplot as plt
plt.hist(hit_number, bins='auto')