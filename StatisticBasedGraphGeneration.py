import argparse
import networkx as nx
from data import *
from stat_rnn import *
from plotter import plotG
import Baseline_statisticBaesd as StatBasdGGMs


def generator(GenerativeModel, G ):
    '''

    :param GenerativeModel:  string indicating the generative model
    :param G: NetworkX trainning grapj
    :return:
    Return the generated geaph
    '''
    G.name = "IDK"
    if GenerativeModel=="BTER":
        GGM = StatBasdGGMs.BTER(G, 1)
    elif GenerativeModel=="ChungLu":
        GGM = StatBasdGGMs.ChungLu(G,1)
    elif GenerativeModel=="ErdosRenyi":
        GGM = StatBasdGGMs.ErdosRenyi(G,1)
    elif GenerativeModel=="SBM":
        GGM = StatBasdGGMs.SBM(G,1)
    else:
        raise Exception("Sorry, the generative model is not implemented")
    GGM._fit()
    G = GGM.generate(1,1)
    G = nx.Graph(G[0])
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if len(G.edges)>0:
        G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
    return [G]



def saveGraphs_to(path, Graphs_List):
    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in Graphs_List]
    np.save( path, graphs_to_writeOnDisk,
                allow_pickle=True)
def graphPloter(listOfNXgraphs, dir, PlotMAXcom=True):

    for i, G in enumerate(listOfNXgraphs):
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if PlotMAXcom:
            if len(G.edges)>0:
                G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
            else:
                print("empty graph")
        plotG(G, file_name=dir+"_"+str(i))




dir = "MMD_statistic_GMMs_result_.1.1/"
Models = ["ChungLu","BTER","ErdosRenyi","SBM"]
# Models = ["BTER"]
Datasets = ["lobster","grid","triangular_grid","ogbg-molbbbp","MUTAG","DD","IMDBBINARY","PTC"]
MMD_eval = []
for model in Models:
    for dataset in Datasets:

        subdir = dataset+"_"+model+"/"
        path_to_save_gen_graphs = dir+subdir+"_generated_set_"
        path_to_save_test_graphs = dir+subdir+"_test_set_"


        if not os.path.exists(dir+subdir):
            os.makedirs(dir+subdir)

        list_adj, _, _ = list_graph_loader(dataset, return_labels=False,limited_to=None)#, _max_list_size=80)
        # split the dataset
        train_list_adj, test_list_adj, _ , _,_,_ = data_split(list_adj,None,None)
        # initialize Model
        random.shuffle(train_list_adj)
        # Generate synthetic graph given a set
        generated_graphs = []
        for graph in train_list_adj[:len(test_list_adj)]:
            New_G = generator(model,nx.from_scipy_sparse_matrix(graph).to_undirected())
            generated_graphs.append(New_G[0])

        # save the graphs
        saveGraphs_to(path_to_save_gen_graphs, generated_graphs)
        saveGraphs_to(path_to_save_test_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
        # plot the generated_graphs
        graphPloter(generated_graphs, dir+subdir)
        #comapre the satistics
        mmd_ = [subdir]
        refSet = [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj]
        refSet = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in refSet]
        mmd_.extend([mmd_eval(generated_graphs, refSet,diam=True)])
        MMD_eval.append(mmd_)
import csv

with open(dir+'MMd_Statisitcs.csv', 'w', newline='\n') as file:
    writer = csv.writer(file)

    writer.writerows([MMD_eval])
