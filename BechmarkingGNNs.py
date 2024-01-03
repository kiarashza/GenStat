import LinkPredictionFramework
import numpy as np
import networkx as nx
import scipy
import random
import csv
import os
from pathlib import Path

# some helper functions
#***************************************
#=======================================
#***************************************
def get_substring_after_last_slash(input_string):
    # Find the last occurrence of "/"
    last_slash_index = input_string.rfind("\\")

    # If no "/" is found, return the whole string
    if last_slash_index == -1:
        return input_string

    # Extract and return the substring after the last "/"
    substring_after_last_slash = input_string[last_slash_index + 1:]
    return substring_after_last_slash

def toCSV(filename, list_):
    with open(filename, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(list_)
def load_graphs(graph_pkl):
    import pickle5 as cp
    graphs = []
    with open(graph_pkl, 'rb') as f:
        while True:
            try:
                g = cp.load(f)
            except:
                break
            graphs.append(g)

    return graphs


def load_graph_list(fname,remove_self=True):

    if fname[-3:]=="pkl":
        glist = load_graphs(fname)
    else:
        with open(fname, "rb") as f:
            glist = np.load(f, allow_pickle=True)

    graph_list =[]
    for G in glist:
        try:
            if type(G)==list:
                if len(G[0])==0:
                    continue
                graph = nx.Graph()
                graph.add_nodes_from(G[0])
                graph.add_edges_from(G[1])
            elif type(G)==nx.classes.graph.Graph:
                graph = G
            elif type(G)==scipy.sparse.csr.csr_matrix:
                graph = nx.from_scipy_sparse_matrix(G)
            else:
                graph = nx.from_numpy_matrix(G)
            if remove_self:
                graph.remove_edges_from(nx.selfloop_edges(graph))
            graph.remove_nodes_from(list(nx.isolates(graph)))
            Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
            graph = graph.subgraph(Gcc[0])
            graph = nx.Graph(graph)
            graph_list.append(graph)
        except Exception as e:
            print("cpould not read a graph")
            print(e)
    return graph_list
def preprocess_graphs(list_of_NXGraph):
    # remove self loops and isolated graphs
    for G in list_of_NXGraph:
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
def extract_DS_name(str, DS_names):
    # extract and return the name of dataset from fiven name
    cleaned_name = ""
    for ds_name in DS_names:
        if ds_name in str:
            return ds_name
    return cleaned_name
# some helper functions

#***************************************
#=======================================
#***************************************



#datasets name
datasets=["lobster","grid","triangular_grid","MUTAG","IMDBBINARY","PTC","DD","ogbg-molbbbp"]
dataset_Original_names=["lobster","triangulargrid","grid","MUTAG","IMDBBINARY","PTC","DD","ogbgmolbbbp"]
# the GNN models
Encoder = ["SGC","GCN","GIN","GAT"]

#........................................ var for reading the generated graphs from dir
# Gen Dir
dir = "C:/git/AAAI/Nips-2023/LastCandidates/" # dir of generated graphs
pattern = "can_"
ref_file_name = "/_Target_Graphs_adj_Test.npy"
gen_graphs_file_name = "/Max_Con_comp_generatedGraphs_adj_Test.npy"
fileName = dir+"LinkPredResult/GenStat/"
#........................................
# BiGG
# dir = "C:/Users/kiarash/Downloads/BiGG.zip-20230910T214917Z-001/bigg_result/data/" # dir of generated graphs
# pattern = ""
# ref_file_name = "/test-graphs.pkl"
# gen_graphs_file_name = "/generated.npy"
# fileName = "C:/git/AAAI/Nips-2023/LDPVAE/GenStatReportingResult/LinkPredResult/BiGG/"
#........................................
# BTER
# dir = "C:/git/temp/" # dir of generated graphs
# pattern = ""
# ref_file_name = "/_test_set_.npy"
# gen_graphs_file_name = "/_generated_set_.npy"
# fileName = "LinkPredResult/BTER/"
#........................................

#either run on the generated graph or original datasets
Benchmark_on_Original_graphs = False
if Benchmark_on_Original_graphs==True:
    fileName = "GenStatReportingResult/LinkPredResult/OriginalGraph/"
#........................................ var for reading the generated graphs from dir



Report = []
datasets_adj = []
if not Benchmark_on_Original_graphs: # read the generated graphs if its necessery/ otherwise the GNN performance on oroginal dataset would be evaluated
    datasets = []

    import glob
    sub_dirs = glob.glob(dir + '*', recursive=True)
    for path in sub_dirs:
        if pattern in path:
            try:
                if  not os.path.isdir(path):
                    continue
                generated = load_graph_list(path+gen_graphs_file_name)[:1000]
                refrence = load_graph_list(path + ref_file_name)[:1000] # max 1000 graph

                print(path)
                random.shuffle(generated)
                generated = generated[:len(refrence)]
                preprocess_graphs(generated)
                datasets.append(path)
                generated = [nx.to_scipy_sparse_matrix(g) for g in generated]
                datasets_adj.append(generated)
            except Exception:
                print(Exception)

# datasets = datasets[-2:]
# datasets_adj = datasets_adj[-2:]
Report = [["dataset Name (DN)","GNN","Cleaned DN","Test_auc",  "Test_acc", "Test_ap", str("conf_mtrx"), "train_auc", "train_acc", "train_ap"]]
AP_AC_forCorrelation = [["dataset","Model","Cleaned DN", "Test_auc",  "Test_acc", "Test_ap"]]
# datasets = datasets[5:]
for encoder in Encoder:
    for d_i,dataset in enumerate(datasets):
        print("dataset:"+dataset)

        the_setting_report =[]
        st_name = extract_DS_name(dataset, dataset_Original_names) + "__" + encoder
        for atpt in range(10):
            if not  Benchmark_on_Original_graphs:
                the_setting_report.append(LinkPredictionFramework.LinkPrediction(datasets_adj[d_i], encoder=encoder, filename=fileName + encoder + "_" + get_substring_after_last_slash(dataset)))
            else:
                the_setting_report.append(LinkPredictionFramework.LinkPrediction(dataset, encoder=encoder, filename=fileName + "_" + encoder + "_" + dataset))
            Report.append([dataset, encoder, st_name])
            Report[-1].extend(the_setting_report[-1])
        AP_AC_ = np.array([x[:3] for x in the_setting_report])
        AP_AC_forCorrelation.append([dataset, encoder, st_name]+list(np.average(AP_AC_,0)))

        Path(fileName+pattern).mkdir(parents=True, exist_ok=True)

        toCSV( fileName+pattern+"BenchmarkGNNsEval.csv",Report)
        toCSV(fileName +pattern+ "AC_AP.csv",AP_AC_forCorrelation)
print(Report)

