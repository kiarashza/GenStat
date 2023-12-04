import numpy as np
from itertools import combinations
import plotter
from pathlib import Path
from data import *

# list(np.load('PVGAErandomGraphs.npy', allow_pickle=True))

rep_dir = "C:/Users/kiarash/Downloads/GRAN.zip-20231204T174344Z-001/GGRAN/GRANMixtureBernoulli_lobster_Kernel_2022-Feb-28-19-21-37_3922529/"
repository_name =  "__gen_adj3000.npy"
dir_to_save = ""

# result = list_graph_loader(dataset)
graphs = list(np.load(rep_dir+repository_name, allow_pickle=True))

random.shuffle(graphs)
for i, G in enumerate(graphs[:20]):
    if type(G)!=nx.classes.graph.Graph:
        G = nx.from_numpy_matrix(G)
    filename = dir_to_save + repository_name

    Path(filename).mkdir(parents=True, exist_ok=True)
    plotter.plotG(G,
                  file_name=filename + "/re_ploted_" + str(i),edge_w=3)
