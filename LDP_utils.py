import numpy as np
import math
from scipy import sparse
import torch
import csv

def RNL(matrix, epsilon):
    """
    matrix: the input matrix, square with  {0,1 }  elements
    elsilon : the LDP parameter; indicate the probablity to switch an element
    Returns matrix, where each element is switch with probability p. (Random neighbour list)
    """
    p = 1/(1+math.e**epsilon)
    original_shape = matrix.shape
    matrix = matrix.reshape(-1)

    mask = np.random.choice([False, True], size=matrix.shape, p=[1-p, p])
    matrix[mask] = 1 - matrix[mask]
    matrix = matrix.reshape(*original_shape)
    return matrix

def DD_RandomMEchanism(EachNode_LocalStatistic, epsilon, delta_f):
    # if statistic have  domian with a single value, it already satisfy any level of LDP
    if len(torch.unique(EachNode_LocalStatistic))==1:
        return EachNode_LocalStatistic
    else:
        return DDP(EachNode_LocalStatistic,epsilon,delta_f)

def DDP(statistic, epsilon, delta_f):
    """

    :param statistic: a torch tensor wich each element represent a node in terms of its local statistics, e.g its node degree, //
    we assumed the local statistic is a number, can be extended to array
    :param elsilon : the LDP parameter; indicate the probablity to switch an element
    :return:
    see for laplace mechanism for the details
    """
    v = delta_f/ epsilon
    noise = np.random.laplace(loc=0.0, scale=v, size=statistic.shape)
    statistic += torch.tensor(noise).to(statistic.device)
     # the satistic cant be  negative
    statistic[statistic<0]=0
    return statistic

def RNL_data_loader(graphType, epsilon_s,dir=""):
    """

    :param graphType: the name of dataset, it should be one the implemented dataset in data.py
    :param epsilon_s: a list of LDP parameters
    :param dir: Disck path
    :return: a data set which perturbed under epsilon-RNL
    """
    import  data
    import networkx as nx
    import os
    if not os.path.exists(dir):
        os.mkdir(dir)
    list_adj, list_x, list_label = data.list_graph_loader(graphType, return_labels=True)
    train_list_adj, test_list_adj, train_list_x, _, train_label, _ = data.data_split(list_adj, list_x, list_label)
    val_adj = train_list_adj[:int(len(test_list_adj) / 2)]
    # ---------------------------------------------------------------------------------------------------------
    train_list_adj = train_list_adj[int(len(val_adj)):]
    # write the test set
    np.save(dir+graphType+'_Test.npy', test_list_adj, allow_pickle=True)
    np.save(dir + graphType + '_Train.npy', train_list_adj, allow_pickle=True)
    from stat_rnn import mmd_eval
    statistics_based_MMD = []
    for epsilon in epsilon_s:
        perturbed_list_adj = []
        # perturb each neibour list in the graph with the LDP parameter
        for adj in train_list_adj:
            matrix = adj.toarray()
            matrix = RNL(matrix, epsilon)
            perturbed_list_adj.append(sparse.csr_matrix(matrix))

        # save the dataset for benchmarks; BiGG

        # statistics_based_MMD.append(mmd_eval([nx.from_scipy_sparse_matrix(adj) for adj in test_list_adj], [nx.from_scipy_sparse_matrix(adj) for adj in perturbed_list_adj], diam=True))
        np.save(dir+graphType+"_"+str(epsilon)+'-LDP_train.npy', perturbed_list_adj, allow_pickle=True)

    # write the test set
    # np.save(dir+graphType+'-LDP_Test.npy', test_list_adj, allow_pickle=True)
    # np.save(dir + graphType + 'train_list_adj-LDP_Train.npy', test_list_adj, allow_pickle=True)
    # statistics_based_MMD = [ [ row] for row in statistics_based_MMD]
    # save the perturbed graph comparion with the ground truth test set in terms of Statistics-based MMD
    # with open(dir+graphType+'_MMDs_ideal.csv', 'w') as f:
    #
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)

        # write.writerows(statistics_based_MMD)

if __name__ == "__main__":
    # example of perturbing adjacency matrix
    # x = np.random.randint(0, high=2, size=(4,4))
    # epsilon = 2
    # RNL(x,epsilon)

    # perturb lobster dataset under [.1,.5, 1, 2,  4,  ]-LDP
    epsilons = [.1,.5]
    for dataset in [ "triangular_grid","IMDBBINARY","PTC","ogbg-molbbbp","DD", "lobster"]:
        RNL_data_loader(dataset,epsilons, "data/LDP/"+dataset+"/") # triangular_grid

    # # example of degree perturbtion
    # delta_f = 1 # sensivity
    # DDP(3,1,delta_f)
    # # example of triangle perturbtion
    # delta_f = 2
    # DDP(3,1,delta_f)
