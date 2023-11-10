from  Synthatic_graph_generator import Synthetic_data
import data
import numpy as np
import scipy



def GraphFeatuer(G):
    return np.concatenate([(G*G>1).sum(-1),(G*G*G>1).sum(-1),np.diagonal((G*G*G).todense()).reshape(-1,1),np.array(G.sum(1)), np.random.rand(G.shape[0],10)],1)

def load_data(dataset,):
    """the method will return adjacncy matrix, node features"""

    if type(dataset)==str:
        adj_list, _, _ = data.list_graph_loader(dataset, return_labels=False)
    else:
        adj_list = dataset
    X_list = []
    for graph in adj_list:
        X_list.append(GraphFeatuer(graph))

    return adj_list, X_list

if __name__ == '__main__':
    # NELL()
    # AMiner()
    load_data("grid")



