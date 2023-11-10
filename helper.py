import pickle

# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


# load a list of graphs
def load_graph_list(fname,is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    # for i in range(len(graph_list)):
    #     edges_with_selfloops = graph_list[i].selfloop_edges()
    #     if len(edges_with_selfloops)>0:
    #         graph_list[i].remove_edges_from(edges_with_selfloops)
    #     if is_real:
    #         graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
    #         graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    #     else:
    #         graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list