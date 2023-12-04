import numpy as np
from itertools import combinations
import plotter
from pathlib import Path
from data import *
# result = list_graph_loader("PTC")
# result = list_graph_loader("ogbg-ppa")
# list(np.load('PVGAErandomGraphs.npy', allow_pickle=True))
# random.seed(10)
# np.random.seed(10)
path = "DataSetSamples/Poster"
datasets = ["IMDBBINARY","lobster", "DD",  "MUTAG", "grid", "triangular_grid", "ogbg-molbbbp", "PTC", "QM9"]
datasets = ["lobster","ogbg-molbbbp"]
for dataset in datasets:
    result = list_graph_loader(dataset)
    import re

    random.shuffle(result[0])
    for i, G in enumerate(result[0][:20]):
        G = nx.from_numpy_matrix(G.toarray())
        filename = path + dataset

        Path(filename).mkdir(parents=True, exist_ok=True)
        plotter.plotG(G, type=dataset ,
                      file_name=filename + "/rand_sa_" + str(i),edge_w=3)
