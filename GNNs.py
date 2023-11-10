

import time
start_time = time.monotonic()
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv as GraphConv
import dgl
from collections import *
from util import *
from scipy.sparse import csr_matrix
from visualization import *
import torch.nn as nn
# from hyperspherical_vae.distributions import VonMisesFisher
# from hyperspherical_vae.distributions import HypersphericalUniform

np.random.seed(0)
random.seed(0)
torch.seed()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch._set_deterministic(True)
import pickle as pickle
import scipy.sparse

# ************************************************************
# encoders
# ************************************************************
#  This class  create a multi-layer of SGC, stacked on each other
class SGC(torch.nn.Module):
    def __init__(self, in_feature, layers=[64], drop_out_rate=0, k=2):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(SGC, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")
        self.ConvLayers = torch.nn.ModuleList(
            dgl.nn.pytorch.conv.SGConv(layers[i], layers[i + 1],k=k,  bias=False) for i in
            range(len(layers) - 1))
        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x, activation= torch.nn.ReLU()):
        for conv_layer in self.ConvLayers:
            x = conv_layer(adj, x)
            x = activation(x)
            # x = self.dropout(x)

        return x
#..................................................................
#  This class  create a multi-layer of GCNs, stacked on each other
class GCN(torch.nn.Module):
    def __init__(self, in_feature, layers=[64], drop_out_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(GCN, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")
        self.ConvLayers = torch.nn.ModuleList(
            GraphConv(layers[i], layers[i + 1], activation=None, bias=False, weight=True) for i in
            range(len(layers) - 1))
        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x, activation= torch.nn.LeakyReLU(0.01)):
        for conv_layer in self.ConvLayers:
            x = conv_layer(adj, x)
            x = activation(x)
            # x = self.dropout(x)

        return x
#..................................................................
#  This class  create a multi-layer of GATs, stacked on each other
class GIN(torch.nn.Module):
    def __init__(self, in_feature, layers=[64],  drop_out_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(GIN, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")

        def mlp(dim, out_dim):
            return nn.Sequential(
                nn.Linear(dim, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, out_dim),
            nn.BatchNorm1d(out_dim)

            )

        self.ConvLayers = torch.nn.ModuleList(dgl.nn.pytorch.GINConv(apply_func=mlp(layers[i], layers[i+1]), aggregator_type='sum',
                                    init_eps=0, learn_eps=False) for i in range(len(layers) - 1))

        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x, activation= torch.nn.LeakyReLU(0.01)):
        for conv_layer in self.ConvLayers:
            x = conv_layer(adj, x)
            x = activation(x)
            # x = self.dropout(x)

        return x
#..................................................................
#  This class  create a multi-layer of GATs, stacked on each other
class GAT(torch.nn.Module):
    def __init__(self, in_feature, layers=[64], NumHeads = 8, drop_out_rate=0):
        """
        :param in_feature: the size of input feature; X.shape()[1]
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        """
        super(GAT, self).__init__()
        layers = [in_feature] + layers
        if len(layers) < 1: raise Exception("sorry, you need at least two layer")
        self.ConvLayers = torch.nn.ModuleList(
            dgl.nn.pytorch.GATConv(layers[i]*NumHeads**(i), layers[i + 1],num_heads=NumHeads,bias=True) for i in
            range(len(layers) - 1))
        self.dropout = torch.nn.Dropout(drop_out_rate)

    def forward(self, adj, x, activation= torch.nn.ELU()):
        for conv_layer in self.ConvLayers:
            x = conv_layer(adj, x)
            x = activation(x.reshape(x.shape[0],-1))
            # x = self.dropout(x)

        return x

# ------------------------------------------------------------------
# Added an Inner Product Decoder for reproducing VGAE Framework
class InnerProductDecoder(torch.nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self):
        # , dropout
        super(InnerProductDecoder, self).__init__()
        # self.dropout = dropout

    def forward(self, z,i_list, j_list):
        z_i = z[i_list]
        z_j = z[j_list]
        adj = (z_i *z_j).sum(-1)
        return adj

# ------------------------------------------------------------------
class MapedInnerProductDecoder(torch.nn.Module):
# we used this decoder in VGAE*, its extention of VGAE in which decoder is definede as DEC(Z) = f(Z)f(Z)^t
    def __init__(self, layers, num_of_relations, in_size, normalize, DropOut_rate):
        """
        :param in_size: the size of input feature; X.shape()[1]
        :param num_of_relations: Number of Latent Layers
        :param layers: a list in which each element determine the size of corresponding GCNN Layer.
        :param normalize: a bool which indicate either use norm layer or not
        """
        super(MapedInnerProductDecoder, self).__init__()
        self.models = torch.nn.ModuleList(
            node_mlp(in_size, layers, normalize, DropOut_rate) for i in range(num_of_relations))

    def forward(self, z, activation = torch.nn.LeakyReLU(0.01)):
        A = []
        for trans_model in self.models:
            tr_z = trans_model(z)
            layer_i = torch.mm(tr_z, tr_z.t(), )
            A.append(layer_i)
        return torch.sum(torch.stack(A), 0)

    def get_edge_features(self, z):
        A = []
        for trans_model in self.models:
            tr_z = trans_model(z)
            layer_i = torch.mm(tr_z, tr_z.t(), )
            A.append(layer_i)
        return A