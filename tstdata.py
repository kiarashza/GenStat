
import torch
from scipy.sparse import *
from  Synthatic_graph_generator import *
# from util import *
import os
import pickle as pkl
import scipy.sparse as sp
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load cora, citeseer and pubmed dataset
def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pkl.load(open("data/Kernel_dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/Kernel_dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G

def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  list_adj = []
  list_x= []
  for G in graphs:
      list_adj.append(nx.adjacency_matrix(G))
      list_x.append(None)
  return list_adj, list_x


class Datasets():
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_adjs,self_for_none, list_Xs, padding =True, Max_num = None):
        """

        :param list_adjs: a list of adjacency in sparse format
        :param list_Xs: a list of node feature matrix
        """
        'Initialization'
        self.paading = padding
        self.list_Xs = list_Xs
        self.list_adjs = list_adjs
        self.toatl_num_of_edges = 0
        self.max_num_nodes = 0
        for i, adj in enumerate(list_adjs):
            list_adjs[i] =  adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            list_adjs[i] += sp.eye(list_adjs[i].shape[0])
            if self.max_num_nodes < adj.shape[0]:
                self.max_num_nodes = adj.shape[0]
            self.toatl_num_of_edges += adj.sum().item()
            # if list_Xs!=None:
            #     self.list_adjs[i], list_Xs[i] = self.permute(list_adjs[i], list_Xs[i])
            # else:
            #     self.list_adjs[i], _ = self.permute(list_adjs[i], None)
        if Max_num!=None:
            self.max_num_nodes = Max_num
        self.processed_Xs = []
        self.processed_adjs = []
        self.num_of_edges = []
        for i in range(self.__len__()):
            a,x,n = self.process(i,self_for_none)
            self.processed_Xs.append(x)
            self.processed_adjs.append(a)
            self.num_of_edges.append(n)
        self.feature_size = self.processed_Xs[0].shape[-1]
        # if list_Xs!=None:
        #     self.feature_size = list_Xs[0].shape[1]
        # else:
        #     self.feature_size = self.max_num_nodes

  def get(self, shuffle= True):
      indexces = list(range(self.__len__()))
      random.shuffle()
      return [self.processed_adjs[i] for i in indexces], [self.processed_Xs[i] for i in indexces]

  def get__(self,from_, to_, self_for_none):
      adj_s = []
      x_s = []
      num_nodes = []
      # padded_to = max([self.list_adjs[i].shape[1] for i in range(from_, to_)])
      # padded_to = 225
      for i in range(from_, to_):
          adj, x, num_node = self.process(i, self_for_none)#, padded_to)
          adj_s.append(adj)
          x_s.append(x)
          num_nodes.append(num_node)
      return adj_s, x_s, num_nodes


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_adjs)

  def process(self,index,self_for_none, padded_to=None,):

      num_nodes = self.list_adjs[index].shape[0]
      if self.paading == True:
          max_num_nodes = self.max_num_nodes if padded_to==None else padded_to
      else:
          max_num_nodes = num_nodes
      adj_padded = lil_matrix((max_num_nodes,max_num_nodes)) # make the size equal to maximum graph
      if max_num_nodes==num_nodes:
          adj_padded = lil_matrix(self.list_adjs[index], dtype=np.int8)
      else:
        adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index][:, :]
      adj_padded -= sp.dia_matrix((adj_padded.diagonal()[np.newaxis, :], [0]), shape=adj_padded.shape)
      if self_for_none:
        adj_padded += sp.eye(max_num_nodes)
      else:
          if max_num_nodes != num_nodes:
              adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
          else:
              adj_padded += sp.eye(num_nodes)
      # adj_padded+= sp.eye(max_num_nodes)




      if self.list_Xs == None:
          # if the feature is not exist we use identical matrix
          X = np.identity( max_num_nodes)
          node_degree = adj_padded.sum(0)
          X = np.concatenate((node_degree.transpose(), X),1 )

      else:
          #ToDo: deal with data with diffrent number of nodes
          X = self.list_Xs[index].toarray()

      # adj_padded, X = self.permute(adj_padded, X)

      # Converting sparse matrix to sparse tensor
      coo = adj_padded.tocoo()
      values = coo.data
      indices = np.vstack((coo.row, coo.col))
      i = torch.LongTensor(indices)
      v = torch.FloatTensor(values)
      shape = coo.shape
      adj_padded = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
      X = torch.tensor(X, dtype=torch.int8)

      return adj_padded.reshape(1,*adj_padded.shape), X.reshape(1, *X.shape), num_nodes

  def permute(self, list_adj, X):
            p = list(range(list_adj.shape[0]))
            np.random.shuffle(p)
            # for i in range(list_adj.shape[0]):
            #     list_adj[:, i] = list_adj[p, i]
            #     X[:, i] = X[p, i]
            # for i in range(list_adj.shape[0]):
            #     list_adj[i, :] = list_adj[i, p]
            #     X[i, :] = X[i, p]
            list_adj[:, :] = list_adj[p, :]
            list_adj[:, :] = list_adj[:, p]
            if X !=None:
                X[:, :] = X[p, :]
                X[:, :] = X[:, p]
            return list_adj , X

  def shuffle(self):
      indx = list(range(len(self.list_adjs)))
      np.random.shuffle(indx)
      if  self.list_Xs !=None:
        self.list_Xs=[self.list_Xs[i] for i in indx]
      self.list_adjs=[self.list_adjs[i] for i in indx]
  def __getitem__(self, index):
        'Generates one sample of data'
        # return self.processed_adjs[index], self.processed_Xs[index],torch.tensor(self.list_adjs[index].todense(), dtype=torch.float32)
        return self.processed_adjs[index], self.processed_Xs[index]
# generate a list of graph
def list_graph_loader( graph_type, _max_list_size=None):
  list_adj = []
  list_x =[]

  if graph_type=="ogbg-molbbbp":
      from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
      from torch.utils.data import DataLoader
      d_name = "ogbg-molbbbp"  # ogbg-molhiv   'ogbg-code2' ogbg-ppa
      dataset = DglGraphPropPredDataset(name=d_name)

      split_idx = dataset.get_idx_split()
      train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=collate_dgl)
      valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
      test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
      list_adj = []
      for graph in test_loader.dataset.dataset.graphs:
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          list_x.append(None)

      for graph in train_loader.dataset.dataset.graphs:
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          list_x.append(None)

      for graph in valid_loader.dataset.dataset.graphs:
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          list_x.append(None)

  if graph_type=="large_grid":
      for i,j in [(100,100)]:
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)
  if graph_type=="grid":
      for i in range(10, 20):
        for j in range(10, 20):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)

  if graph_type=="IMDbMulti":
      list_adj = pkl.load(open("data/IMDbMulti/IMDBMulti.p",'rb'))
      list_x= [None for x in list_adj]
  if graph_type=="one_grid":
        list_adj.append(nx.adjacency_matrix(grid(8, 8)))
        list_x.append(None)
  if graph_type=="small_grid":
      for i in range(4, 8):
        for j in range(4, 8):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)
  elif graph_type=="community":
      for i in range(30, 81):
        for j in range(30,81):
            list_adj.append(nx.adjacency_matrix(n_community([i, j], p_inter=0.3, p_intera=0.05)))
            list_x.append(None)

  elif graph_type == 'large_lobster':
      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 1000
      max_node = 10000
      max_edge = 0
      mean_node = 5000
      num_graphs = 100
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1

  elif graph_type == 'lobster':
      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 10
      max_node = 100
      max_edge = 0
      mean_node = 80
      num_graphs = 100
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1
  elif graph_type == "cora":
      import input_data
      list_adj, list_x, _,_,_ = input_data.load_data(graph_type)
      list_adj = [list_adj]
      list_x = [list_x]
  elif graph_type == "ACM":
      import input_data
      list_adj, list_x, _,_,_ = input_data.load_data(graph_type)
      list_adj = [list_adj]
      list_x = [list_x]
  elif graph_type == 'ego':
      _, _, G = Graph_load(dataset='citeseer')
      # G = max(nx.connected_component_subgraphs(G), key=len)
      G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
      G = nx.convert_node_labels_to_integers(G)
      graphs = []
      for i in range(G.number_of_nodes()):
          G_ego = nx.ego_graph(G, i, radius=3)
          if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
              graphs.append(G_ego)
              list_adj.append(nx.adjacency_matrix(G_ego))
              list_x.append(None)



  elif graph_type == 'DD':
    list_adj, list_x  = graph_load_batch(
        "data/Kernel_dataset/",
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
    # args.max_prev_node = 230

  # for j in len(list_adj):
  #       p = list(range(list_adj[j].shape[0]))
  #       np.random.shuffle(p)
  #       for i in range(list_adj[j].shape[0]):
  #           list_adj[j][:, i] = list_adj[j][p, i]
  #       for i in range(list_adj[j].shape[0]):
  #           list_adj[j][i, :] = list_adj[j][i, p]

  temp = list(zip(list_adj, list_x))
  # random.shuffle(temp)
  list_adj, list_x = zip(*temp)

  if _max_list_size!=None:
      list_adj = list(list_adj[:_max_list_size])
      list_x = list(list_x[:_max_list_size])
  else:
      list_adj = list(list_adj)
      list_x = list(list_x)
  return list_adj, list_x

def data_split(graph_lis):
    random.seed(123)
    random.shuffle(graph_lis)
    graph_test_len = len(graph_lis)

    graph_train = graph_lis[0:int(0.8 * graph_test_len)]  # train
    # graph_validate = graph_lis[0:int(0.2 * graph_test_len)]  # validate
    graph_test = graph_lis[int(0.8 * graph_test_len):]  # test on a hold out test set
    return  graph_train, graph_test

# list_adj, list_x = list_graph_loader("grid")
# list_graph = Datasets(list_adj,self_for_none, None)

if __name__ == '__main__':
    result = list_graph_loader( "large_grid")
    for G in result[0]:
        import plotter

        G = nx.from_numpy_matrix(G.toarray())
        plotter.plotG(G,"DD")
#     # Parameters
#     params = {'batch_size': 64,
#               'shuffle': True,
#               'num_workers': 1}
#     max_epochs = 100
#
#     training_generator = torch.utils.data.DataLoader(list_graph, **params)
#
#     for iter, (adj_s, x_s) in enumerate(training_generator):
#         pass

def permute(list_adj, X):
    for i, _ in enumerate(list_adj):
        p = list(range(list_adj[i].shape[0]))
        np.random.shuffle(p)

        list_adj[i][:, :] = list_adj[i][p, :]
        list_adj[i][:, :] = list_adj[i][:, p]
        list_adj[i].eliminate_zeros()
        if X != None:
            X[:, :] = X[p, :]
            X[:, :] = X[:, p]
    return list_adj, X

