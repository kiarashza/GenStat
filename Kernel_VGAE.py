# Unexpected error: <class 'OSError'>
# Unexpected error: <class 'OSError'>
# Unexpected error: <class 'OSError'>
# Unexpected error: <class 'ZeroDivisionError'>
# degree 0.07535966601684341 clustering 0.048087770412737596 orbits -1

import torch as torch
import torch.nn.functional as F
import numpy as np
from mask_test_edges import mask_test_edges, roc_auc_estimator,roc_auc_estimator_onGraphList
from input_data import load_data
import scipy.sparse as sp
import graph_statistics as GS
import plotter
import networkx as nx
import os
import argparse
from util import *
from data import *
import pickle
import math

import random as random
import time
import timeit
import dgl

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

subgraphSize = None
keepThebest = False
parser = argparse.ArgumentParser(description='Kernel VGAE')
parser.add_argument('-e', dest="epoch_number" , default=5000, help="Number of Epochs")
parser.add_argument('-v', dest="Vis_step", default=390, help="model learning rate")

parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.0001, help="model learning rate") # for RNN decoder use 0.0001
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1, help="the rate of negative samples which shold be used in each epoch; by default negative sampling wont use")
parser.add_argument('-dataset', dest="dataset", default="small_lobster", help="possible choices are:  ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")#citeceer: ego; DD:protein

parser.add_argument('-NofCom', dest="num_of_comunities", default=64, help="Number of comunites")
parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
parser.add_argument('-graph_save_path', dest="graph_save_path", default="develop_lobster_kernel/", help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature" , default=True, help="either use features or identity matrix")
parser.add_argument('-Split', dest="split_the_data_to_train_test" , default=True, help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-PATH', dest="PATH" , default="model", help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder" , default="GraphTransformerDecoder", help="the decoder type,GraphTransformerDecoder, SBMdecoder, FC_InnerDOTdecoder, GRAPHdecoder,FCdecoder,")
parser.add_argument('-batchSize', dest="batchSize" , default=100, help="the size of each batch")
parser.add_argument('-UseGPU', dest="UseGPU" , default=True, help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model" , default="kernel", help="sbm, kipf or kernel")
parser.add_argument('-device', dest="device" , default="cuda:0", help="Which device should be used")
parser.add_argument('-task', dest="task" , default="graphGeneration", help="linkPrediction or graphGeneration")


args = parser.parse_args()
# torch.autograd.set_detect_anomaly(True)


# **********************************************************************
# setting
print("KernelVGAE SETING: "+str(args))
PATH = args.PATH # the dir to save the with the best performance on validation data
visulizer_step = args.Vis_step
redraw = args.redraw
device = args.device
task = args.task
epoch_number = args.epoch_number
lr = args.lr
negative_sampling_rate = args.negative_sampling_rate
hidden_1 = args.num_of_comunities  # ?????????????? naming
decoder_type = args.decoder
hidden_2 =  hidden_1 # number of comunities;
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
mini_batch_size = args.batchSize
use_gpu=args.UseGPU
use_feature = args.use_feature
save_embeddings_to_file = args.save_embeddings_to_file
graph_save_path = args.graph_save_path
split_the_data_to_train_test = args.split_the_data_to_train_test

kernl_type = []

if args.model == "kernel_tri":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "tri", "square"]
    alpha = [1, 1, 1, 1, 1, 1e-06, 1e-06, 0, 0, .001, .001 * 20]
    step_num = 5
# alpha= [1, 1, 1, 1, 1e-06, 1e-06, 0, 0,.001]
if args.model == "kernel":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist"]
    if dataset=="large_grid":
        alpha = [10, 10, 10, 10, 10, .5e-07, .5e-07, .001, .001 * 20]
    if dataset in {"grid" , "small_grid", "one_grid"} :
        alpha = [1, 1, 1, 1, 1, 1e-06, 1e-06, .001, .001 * 20]
    elif dataset == "DD":
        alpha= [10,10, 10, 10, 10, .2*1e-06, .2*1e-06,.001,.001*20] #DD
    elif dataset == "lobster" or  dataset == "small_lobster":
        alpha= [1,1, 1, 1, 1, 1e-06, 1e-06,.001 ,.001 * 20]
    elif dataset == "ogbg-molbbbp":
        alpha = [1, 1, 1, 1, 1, 1e-06, 1e-06, .001, .001 * 20]
    elif dataset == "IMDbMulti":
        alpha = [1, 1, 1, 1, 1, 1e-06, 1e-06, .001, .001 * 20]
        # alpha = [10, 10, 10, 10, 10, 1e-05, 1e-05, .001, .001 * 20]

    # alpha= [1,1, 1, 1, 1, 1e-06, 1e-06,.001 ,.001 * 20] #GRID
    # alpha= [2,2, 2, 2, 2, .51e-06, .51e-06,.001,.001* 20] #GRID_sbm

    #alpha = [1, 1, 1, 1, 1, 2e-08, 2e-08, 0.001, 0.008] #SBM_Lobster
    # alpha = [1,1, 1, 1, 1, 1e-06, 1e-06,.001 ,.001 * 20]  # DD

    #alpha = [10, 10, 50, 100, 100, 1e-09, 1e-09, 0.02, 0.004]  # SBM_Lobster
    #alpha= [1,1, 1, 1, 1, 1e-06, 1e-06,.001 ,.001 * 20]  # SBM_Lobster

    # alpha = [1, 1, 1, 1, 1, 2e-08, 2e-08, 0.001, 0.008] #SBM_Lobster
    # alpha = [1,1, 1, 1, 1, 1e-06, 1e-06,.001 ,.001 * 20]  # DD

    # alpha = [10, 10, 50, 100, 100, 1e-09, 1e-09, 0.02, 0.004]  # SBM_Lobster
    # alpha= [1,1, 1, 1, 1, 1e-06, 1e-06,.001 ,.001 * 20]  # SBM_Lobster

    step_num = 5
if args.model == "kipf":
    alpha= [ .001,.001*20]
    step_num = 0
if args.model == "sbm":
    alpha= [ .001,.001]
    step_num = 0
kernl_type_eval = ["trans_matrix", "in_degree_dist", "out_degree_dist"]
#     alpha= [10,10, 10, 10, 10, .2*1e-06, .2*1e-06,.001,.001*20] #DD
alpha_eval = [1,1, 1, 1, 1, .1e-06, .1e-06,.001 ,.001 * 20] #GRID
# alpha= [2,2, 2, 2, 2, .51e-06, .51e-06,.001,.001* 20] #GRID
step_num_eval = 5
print("kernl_type:"+str(kernl_type))
print("alpha: "+str(alpha) +" num_step:"+str(step_num))

bin_center = torch.tensor([[x / 10000] for x in range(0, 1000, 1)])
bin_width = torch.tensor([[9000] for x in range(0, 1000, 1)])# with is propertion to revese of this value;

device = torch.device(device if torch.cuda.is_available() and use_gpu else "cpu")
print("the selected device is :", device)

# setting the plots legend
functions= ["Accuracy", "loss", "AUC"]
functions.extend(["Kernel"+str(i) for i in range(step_num)])
functions.extend(kernl_type[1:])
functions.append("Binary_Cross_Entropy")
functions.append("KL-D")
#========================================================================
functions_eval = ["Accuracy", "loss", "AUC"]
functions_eval.extend(["Kernel"+str(i) for i in range(step_num_eval)])
functions_eval.extend(kernl_type_eval[1:])
functions_eval.append("Binary_Cross_Entropy")
functions_eval.append("KL-D")
#========================================================================



pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log",functions=functions)

synthesis_graphs = {"grid","small_lobster","small_grid", "community", "lobster", "ego", "one_grid"}
if dataset in synthesis_graphs:
    split_the_data_to_train_test = False


# **********************************************************************
class kernelGVAE(torch.nn.Module):
    def __init__(self, in_feature_dim, hidden1,  latent_size, ker, decoder, encoder_fcc_dim = None ):
        super(kernelGVAE, self).__init__()
        # self.first_conv_layer = GraphConvNN(in_feature_dim, hidden1)
        # self.second_conv_layer = GraphConvNN(hidden1, hidden1)
        # self.stochastic_mean_layer = GraphConvNN(hidden1, latent_size)
        # self.stochastic_log_std_layer = GraphConvNN(hidden1, latent_size)
        self.first_conv_layer = dgl.nn.pytorch.conv.GraphConv(in_feature_dim, hidden1, activation=None, bias=True,
                                                              weight=True)
        self.second_conv_layer = dgl.nn.pytorch.conv.GraphConv(hidden1, hidden1, activation=None, bias=True, weight=True)
        self.stochastic_mean_layer = dgl.nn.pytorch.conv.GraphConv(hidden1, latent_size, activation=None, bias=True,
                                                                   weight=True)
        self.stochastic_log_std_layer = dgl.nn.pytorch.conv.GraphConv(hidden1, latent_size, activation=None, bias=True,
                                                                      weight=True)
        self.kernel = ker #TODO: bin and width whould be determined if kernel is his

        # self.reset_parameters()
        self.Drop = torch.nn.Dropout(0)
        self.Drop = torch.nn.Dropout(0)
        self.latent_dim = latent_size
        self.mlp = None
        self.decode = decoder

        if None !=encoder_fcc_dim:

            self.fnn =node_mlp(hidden1, encoder_fcc_dim)
            self.stochastic_mean_layer = node_mlp(encoder_fcc_dim[-1], [latent_size])
            self.stochastic_log_std_layer = node_mlp(encoder_fcc_dim[-1], [latent_size])

    def forward(self, graph, features, num_node,batchSize , subgraphs_indexes):
        """

        :param graph: normalized adjacency matrix of graph
        :param features: normalized node feature matrix
        :return:
        """
        samples, mean, log_std = self.encode( graph, features, batchSize)
        if type(self.decode)==GRAPHITdecoder:
            reconstructed_adj_logit = self.decode(samples,features)
        if type(self.decode)==graphitDecoder:
            reconstructed_adj_logit = self.decode(samples,features)
        elif type(self.decode)==RNNDecoder:
            reconstructed_adj_logit = self.decode(samples,num_node)
        else:
            reconstructed_adj_logit = self.decode(samples, subgraphs_indexes)
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)


        mask = torch.zeros(reconstructed_adj.shape)

        # removing the effect of none existing nodes
        for i in range(reconstructed_adj.shape[0]):
            reconstructed_adj_logit[i, :, num_node[i]:] = -100
            reconstructed_adj_logit[i, num_node[i]:, :] = -100
            mask[i, :num_node[i], :num_node[i]] = 1
            mean[i,num_node[i]:, :]=  0
            log_std[i,num_node[i]:, :]=  0

        reconstructed_adj = reconstructed_adj * mask.to(device)
        kernel_value = self.kernel(reconstructed_adj)
        # reconstructed_adj_logit  = reconstructed_adj_logit + mask_logit
        return reconstructed_adj, samples, mean, log_std, kernel_value, reconstructed_adj_logit

    def encode(self, graph, features, batchSize, activation= torch.nn.LeakyReLU(0.01)):
        h = self.first_conv_layer(graph, features)
        h = self.Drop(h)
        h= activation(h)
        h = self.second_conv_layer(graph, h)
        h = activation(h)
        if type(self.stochastic_mean_layer) ==GraphConvNN:
            mean = self.stochastic_mean_layer(graph, h)
            log_std = self.stochastic_log_std_layer(graph, h)
        else:
            h = self.fnn(h, activation)
            mean = self.stochastic_mean_layer(h,activation = lambda x:x)
            log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)

        mean = mean.reshape(*batchSize, -1)
        log_std = log_std.reshape(*batchSize, -1)

        sample = self.reparameterize(mean, log_std, node_num)


        return sample, mean, log_std


    def reparameterize(self, mean, log_std, node_num):
        # std = torch.exp(log_std)
        # eps = torch.randn_like(std)
        # return eps.mul(std).add(mean)

        var = torch.exp(log_std).pow(2)
        eps = torch.randn_like(var)
        sample = eps.mul(var).add(mean)

        for i, node_size in enumerate(node_num):
            sample[i][node_size:,:]=0
        return sample





class Histogram(torch.nn.Module):
    # this is a soft histograam Function.
    #for deails check section "3.2. The Learnable Histogram Layer" of
    # "Learnable Histogram: Statistical Context Features for Deep Neural Networks"
    def __init__(self, bin_width = None, bin_centers = None):
        super(Histogram, self).__init__()
        self.bin_width = bin_width.to(device)
        self.bin_center = bin_centers.to(device)
        if self.bin_width == None:
            self.prism()
        else:
            self.bin_num = self.bin_width.shape[0]

    def forward(self, vec):
        #REceive a vector and return the soft histogram

        #comparing each element with each of the bin center
        score_vec = vec.view(vec.shape[0],1, vec.shape[1], ) - self.bin_center
        # score_vec = vec-self.bin_center
        score_vec = 1-torch.abs(score_vec)*self.bin_width
        score_vec = torch.relu(score_vec)
        return score_vec.sum(2)

    def prism(self):
        pass
# decoder
#============================================================================
class RNNDecoder(torch.nn.Module):
    def __init__(self, input, h_size = 1024):
        super(RNNDecoder, self).__init__()
        self.rnn = torch.nn.GRU(input, h_size,num_layers=3 ,batch_first=True)
        self.decoder = SBMdecoder_(h_size)
        self.input = input
    def forward(self, z, node_num):
        z = torch.nn.utils.rnn.pack_padded_sequence(z, batch_first=True, lengths=node_num, enforce_sorted=False)
        z,_ = self.rnn(z)
        z,_ =torch.nn.utils.rnn.pad_packed_sequence(z, batch_first=True)
        return self.decoder(z)


class GRAPHdecoder(torch.nn.Module):
    def __init__(self, input, layers= [128,128,]):
        super(GRAPHdecoder, self).__init__()
        latent_layer = [input] + layers
        self.layers = torch.nn.ModuleList(GRAPHdecoder_layer(latent_layer[i], [latent_layer[i+1]]) for i in range(len(latent_layer)-1))
        self.decoder = SBMdecoder_(latent_layer[-1],None)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)

        return self.decoder(z)

class GRAPHdecoder_layer(torch.nn.Module):
    def __init__(self,input, MLP_layers= [512, 512,512], graoh_layers = [512], max_num_node=None, type = "sum"):
        super(GRAPHdecoder_layer, self).__init__()
        self.type= type
        if self.type == sum:
            self.graph_rep = node_mlp(input, 2*MLP_layers  , dropout_rate=.0, normalize=True)
        else:
            graoh_layers = [max_num_node*input] + graoh_layers
            self.graph_rep = torch.nn.ModuleList([torch.nn.Linear(graoh_layers[i],graoh_layers[i+1]) for i in range(len(graoh_layers))])
        self.node_rep = node_mlp(input+MLP_layers[-1],2*MLP_layers, dropout_rate=.0, normalize=True)

    def forward(self, z, activation=lambda x: x):
        if self.type=="sum":
            G = self.graph_rep(z)
            G = G.sum(-2)
            G= torch.tanh(G)
        else:
            G = self.graph_rep(z.reshape(-1))

        G = torch.unsqueeze(G, 1).repeat(1, z.shape[-2], 1)
        z = self.node_rep(torch.cat((z,G),2))
        return z
        # return activation(torch.matmul(z,z.permute(0,2,1)))


#--------------------------------------------------------------------------------

class GRAPHITdecoder(torch.nn.Module):
    def __init__(self,latent_size, X_dim, latent_layers= [128,128]):
        super(GRAPHITdecoder, self).__init__()
        latent_layers = [latent_size] + latent_layers
        self.decoders = torch.nn.ModuleList(SBMdecoder_(i, None) for i in latent_layers[:-1])
        self.decoder = SBMdecoder_(latent_layers[-1],None)
        self.gnns = torch.nn.ModuleList(GraphConvNN(X_dim, latent_layers[i+1]) for i in range(len(latent_layers)-1))

    def forward(self, z, x):
        for i in range(len(self.decoders)):
            adj = self.decoders[i](z)#,activation=torch.sigmoid)
            # adj = adj +torch.eye(adj[0].shape[0])
            adj = torch.tanh(adj)
            # z = self.gnns[i](adj, torch.cat((z,x),2))
            z = self.gnns[i](adj,x)
        adj = self.decoder(z)
        return adj

class graphitLayer(torch.nn.Module):
    def __init__(self, X_dim, Z_dim):
        super(graphitLayer, self).__init__()
        self.GCN = GraphConvNN(X_dim, Z_dim)

    def forward(self, Z, X):
        #A = torch.matmul(Z,Z.permute(0, 2, 1))/torch.sqrt(torch.sum(Z*Z,[-1,-2])).reshape(Z.shape[0],1,1)+1
        # A = torch.matmul(Z/(torch.norm(Z, dim=0, keepdim=True,p=2)+.001), Z.permute(0, 2, 1))+1
        # A = torch.relu(torch.matmul(Z , Z.permute(0, 2, 1)))

        #A = torch.matmul(Z / (torch.norm(Z, dim=2, keepdim=True, p=2) + .001), (Z / (torch.norm(Z, dim=2, keepdim=True, p=2) + .001)).permute(0, 2, 1)) + 1
        A = torch.relu(torch.matmul(Z, (Z.permute(0, 2, 1) )))

        Z = self.GCN(A,Z)
        return Z

class graphitDecoder(torch.nn.Module):
    def __init__(self, X_dim, Z_dim, layers= [1024]):
        super(graphitDecoder, self).__init__()
        layers_dim = [Z_dim]+layers
        self.layers =  torch.nn.ModuleList([graphitLayer(layers_dim[i],layers_dim[i+1]) for i in range(len(layers_dim)-1)])
        self.GCN1 = GraphConvNN(Z_dim, Z_dim)
        self.GCN2 = GraphConvNN(X_dim, Z_dim)
        self.GCN3 = GraphConvNN(Z_dim, Z_dim)
    def forward(self, Z, X):
        # for i,layer in enumerate(self.layers):
        #     Z = layer(Z,X)
        #     # if i!=(len(self.layers)-1):
        #     #     Z = torch.relu(Z)
        # return torch.matmul(Z,Z.permute(0, 2, 1))
        A = torch.matmul((Z+1) / (torch.norm((Z+1), dim=2, keepdim=True, p=2) + .001), ((Z+1) / (torch.norm((Z+1), dim=2, keepdim=True, p=2) + .001)).permute(0, 2, 1))
        A = torch.relu(torch.matmul(Z, Z.permute(0, 2, 1)))


        recon_1 = Z / (torch.norm((Z), dim=2, keepdim=True, p=2))
        recon_2 = torch.ones_like(recon_1)
        recon_2 /= torch.sqrt(torch.sum(recon_2, axis = 2, keepdim=True))


        A = torch.matmul(recon_1, recon_1.permute(0, 2, 1)) + torch.matmul(recon_2, recon_2.permute(0, 2, 1))
        Z1 = torch.relu(self.GCN1(A,Z)) + torch.relu(self.GCN2(A,X))
        # Z1 = torch.relu(Z1)
        Z2 = self.GCN3(A,Z1)
        # Z2 = torch.sigmoid(Z2)
        Z = .5*Z+.5*Z2
        return torch.matmul(Z, Z.permute(0, 2, 1))


class smartFNN(torch.nn.Module):
    def __init__(self, inputSize, graph_latentLayer=[10048],FCLayer=[4048, 2048] ,  nodeTransferLayer = [4048, 1024, 1]):
        super(smartFNN, self).__init__()
        self.graphTrns = Graph_mlp(inputSize,  graph_latentLayer)
        FCLayer =  [graph_latentLayer[-1]]+ FCLayer
        self.FC = torch.nn.ModuleList([torch.nn.Linear(FCLayer[i], FCLayer[i + 1]) for i in range(len(FCLayer) - 1)])

        nodeTransferLayer =  [2* inputSize+ FCLayer[-1]] + nodeTransferLayer
        self.graphTranferlayers = torch.nn.ModuleList([torch.nn.Linear(nodeTransferLayer[i], nodeTransferLayer[i + 1]) for i in range(len(nodeTransferLayer) - 1)])

    def forward(self, in_tensor,subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.01)):
        graphEm = self.graphTrns(in_tensor, activation)

        for layer in self.FC:
            graphEm = layer(graphEm)
            graphEm = activation(graphEm)

        #merging the embedding
        z = torch.cat((torch.unsqueeze(in_tensor, 1).repeat(1, in_tensor.shape[1], 1, 1), torch.unsqueeze(in_tensor, 2).repeat(1, 1, in_tensor.shape[1], 1)),-1)
        z = torch.cat((z, graphEm.reshape([z.shape[0],1,1,graphEm.shape[-1]]).repeat(1,z.shape[1],z.shape[1],1)),-1)

        for i, layer in enumerate(self.graphTranferlayers):
            z = layer(z)
            if i<(len(self.graphTranferlayers)-1):
                z = activation(z)
        return z.reshape(z.shape[:-1])

# class smartFNN(torch.nn.Module):
#     def __init__(self, inputSize, graph_latentLayer=[1024], FCLayer = [1024, 256, 1]):
#         super(smartFNN, self).__init__()
#         self.graphTrns = Graph_mlp(inputSize,  graph_latentLayer)
#         FCLayer =  [2* inputSize+ graph_latentLayer[-1]] + FCLayer
#         self.graphTranferlayers = torch.nn.ModuleList([torch.nn.Linear(FCLayer[i], FCLayer[i + 1]) for i in range(len(FCLayer) - 1)])
#
#     def forward(self, in_tensor,subgraphs_indexes=None, activation= torch.relu):
#         graphEm = self.graphTrns(in_tensor, activation)
#
#         z = torch.cat((torch.unsqueeze(in_tensor, 1).repeat(1, 100, 1, 1), torch.unsqueeze(in_tensor, 2).repeat(1, 1, 100, 1)),-1)
#
#         z = torch.cat((z, graphEm.reshape([1,1,1,-1]).repeat(*z.shape[:-1],1)),-1)
#         for i, layer in enumerate(self.graphTranferlayers):
#             z = layer(z)
#             if i<(len(self.graphTranferlayers)-1):
#                 z = activation(z)
#         return z.reshape(z.shape[:-1])

class SmarterSBMdecoder_(torch.nn.Module):
    def __init__(self,inputSize,latent_size=512, MLP_layers= [128 ], ):
        super(SmarterSBMdecoder_, self).__init__()
        self.mlp = None
        if None !=MLP_layers:
            self.mlp = node_mlp(inputSize,[256, latent_size], dropout_rate=.0, normalize=False)

        self.graphTrns = Graph_mlp(latent_size, [ 1028])
        # layers= [latent_size, latent_size, latent_size*latent_size]
        layers = [1028, latent_size * latent_size]

        self.graphTranferlayers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.lamda =torch.nn.Parameter(torch.Tensor(latent_size, latent_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)


    def forward(self, z, subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.01)):
        # return torch.mm(z1,z2.t())
        if self.mlp != None:
            z = self.mlp(z, activation)

        graphEm = self.graphTrns(z,activation)
        for i, layer in enumerate(self.graphTranferlayers):
            graphEm = layer(graphEm)
            if i!=len(self.graphTranferlayers)-1:
                graphEm = activation(graphEm)
        return torch.matmul(torch.matmul(z, graphEm.reshape(-1, z.shape[-1], z.shape[-1])), z.permute(0, 2, 1))
        return torch.matmul(torch.matmul(z, graphEm ), z.permute(0, 2, 1))



        return torch.matmul(torch.matmul(z, graphEm.reshape(-1, z.shape[-1], z.shape[-1])), z.permute(0, 2, 1))



class SBMdecoder_(torch.nn.Module):
    def __init__(self,latent_size, lambda_dim = 256, MLP_layers= [256,1024 , 1024]):
        super(SBMdecoder_, self).__init__()
        self.lamda =torch.nn.Parameter(torch.Tensor(lambda_dim, lambda_dim))
        self.mlp = None
        if None !=MLP_layers:
            self.mlp = node_mlp(latent_size, MLP_layers+[lambda_dim], dropout_rate=.0, normalize=False)
        self.reset_parameters()
    def forward(self, z, subgraph_indexea=None,activation=torch.nn.LeakyReLU(0.01)):
        # return torch.mm(z1,z2.t())
        if self.mlp != None:
            z = self.mlp(z,activation =activation,  applyActOnTheLastLyr=False)
        return torch.matmul(torch.matmul(z, self.lamda), z.permute(0, 2, 1))
        # return activation(torch.matmul(z,z.permute(0,2,1)))

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)

class FCdecoder(torch.nn.Module):
    def __init__(self,input,output,layer=[512,512,512]):
        super(FCdecoder,self).__init__()
        layer = [input]+layer + [output]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layer[i],layer[i+1]) for i in range(len(layer)-1)])
    # def forward(self,Z):
    #     shape = Z.shape
    #     z = Z.reshape(shape[0],-1)
    #     for i in range(len(self.layers)):
    #         z  = self.layers[i](z)
    #         z = torch.tanh(z)
    #     # Z = torch.sigmoid(Z)
    # return z.reshape(shape[0], shape[-2], shape[-2])
    def forward(self, in_tensor, activation=torch.nn.ReLU()):
        h = in_tensor.reshape(in_tensor.shape[0],-1)
        for i in range(len(self.layers)):
            # if self.norm_layers != None:
            #     if len(h.shape) == 2:
            #         h = self.norm_layers[i](h)
            #     else:
            #         shape = h.shape
            #         h = h.reshape(-1, h.shape[-1])
            #         h = self.norm_layers[i](h)
            #         h = h.reshape(shape)
            # h = self.dropout(h)
            h = self.layers[i](h)
            if ((i!=len(self.layers)-1)):
                h = activation(h)
        return h.reshape(in_tensor.shape[0], in_tensor.shape[-2], in_tensor.shape[-2])


# class FC_InnerDOTdecoder(torch.nn.Module):
#     def __init__(self,input,output,node_dim,laten_size ,layer=[1024,1024,1024]):
#         super(FC_InnerDOTdecoder,self).__init__()
#         self.lamda = torch.nn.Parameter(torch.Tensor(laten_size, laten_size))
#         layer = [input]+layer + [output]
#         self.layers = torch.nn.ModuleList([torch.nn.Linear(layer[i],layer[i+1]) for i in range(len(layer)-1)])
#         self.node_transfer = node_mlp(node_dim, [node_dim, laten_size], dropout_rate=.0, normalize=False)
#         self.reset_parameters()
#     # def forward(self,Z):
#     #     shape = Z.shape
#     #     z = Z.reshape(shape[0],-1)
#     #     for i in range(len(self.layers)):
#     #         z  = self.layers[i](z)
#     #         z = torch.tanh(z)
#     #     # Z = torch.sigmoid(Z)
#     # return z.reshape(shape[0], shape[-2], shape[-2])
#     def forward(self, in_tensor, activation=torch.nn.ReLU()):
#         h = in_tensor.reshape(in_tensor.shape[0],-1)
#         for i in range(len(self.layers)):
#             # if self.norm_layers != None:
#             #     if len(h.shape) == 2:
#             #         h = self.norm_layers[i](h)
#             #     else:
#             #         shape = h.shape
#             #         h = h.reshape(-1, h.shape[-1])
#             #         h = self.norm_layers[i](h)
#             #         h = h.reshape(shape)
#             # h = self.dropout(h)
#             h = self.layers[i](h)
#             if ((i!=len(self.layers))):
#               h = activation(h)
#         h = h.reshape(in_tensor.shape[0], in_tensor.shape[1],-1)
#         h = self.node_transfer(h)
#         return torch.matmul(torch.matmul(h,self.lamda), h.permute(0, 2, 1))
#
#     def reset_parameters(self):
#         self.lamda = torch.nn.init.xavier_uniform_(self.lamda)
class GraphTransformerDecoder(torch.nn.Module):
    def __init__(self,input,lambdaDim,SubGraphNodeNum, layers= [128 ]):
        super(GraphTransformerDecoder, self).__init__()
        self.lamda =torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
        self.SubGraphNodeNum = SubGraphNodeNum
        layers = [4096] + [2048, 1028]+[lambdaDim*SubGraphNodeNum]
        self.graphTrns = Graph_mlp(input, [ 4096])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.reset_parameters()

    def forward(self, in_tensor, subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.001)):
        if subgraphs_indexes!=None:
            z = []
            for i in range(len(subgraphs_indexes)):
                z.append(in_tensor[i][subgraphs_indexes[i]])
            in_tensor = torch.stack(z)
            del z

        Graph_num = in_tensor.shape[0]
        h = self.graphTrns(in_tensor, activation)
        del in_tensor
        for i in range(len(self.layers)):
            h = self.layers[i](h)
            if i !=len((self.layers))-1:
                h = activation(h)
        h = h.reshape(Graph_num, self.SubGraphNodeNum, -1)

        adj_list= torch.matmul(torch.matmul(h, self.lamda),h.permute(0,2,1))
        return adj_list
        # if subgraphs_indexes==None:
        #     adj_list= torch.matmul(torch.matmul(h, self.lamda),h.permute(0,2,1))
        #     return adj_list
        # else:
        #     adj_list = []
        #     for i in range(h.shape[0]):
        #         adj_list.append(torch.matmul(torch.matmul(h[i][subgraphs_indexes[i]], self.lamda),h[i][subgraphs_indexes[i]].permute( 1, 0)))
        #     return torch.stack(adj_list)

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)

class GraphTransformerDecoder2(torch.nn.Module):
    def __init__(self,input,lambdaDim,SubGraphNodeNum, layers= [128 ]):
        super(GraphTransformerDecoder2, self).__init__()
        self.SubGraphNodeNum = SubGraphNodeNum
        layers = [4096] + [2048, 1028]+[lambdaDim*SubGraphNodeNum]
        self.graphTrns = Graph_mlp(input, [ 4096])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        linkPredictionLayers = [2 * lambdaDim,256 ,1]

        self.nodePredict = torch.nn.ModuleList(
            [torch.nn.Linear(linkPredictionLayers[i], linkPredictionLayers[i + 1]) for i in
             range(len(linkPredictionLayers) - 1)])


    def forward(self, in_tensor, subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.01)):
        if subgraphs_indexes!=None:
            z = []
            for i in range(len(subgraphs_indexes)):
                z.append(in_tensor[i][subgraphs_indexes[i]])
            in_tensor = torch.stack(z)
            del z

        Graph_num = in_tensor.shape[0]
        h = self.graphTrns(in_tensor, activation)
        del in_tensor
        for i in range(len(self.layers)):
            h = self.layers[i](h)
            if i !=len((self.layers))-1:
                h = activation(h)
        h = h.reshape(Graph_num, self.SubGraphNodeNum, -1)

        #merging the embedding
        h = torch.cat((torch.unsqueeze(h, 1).repeat(1, h.shape[1], 1, 1), torch.unsqueeze(h, 2).repeat(1, 1, h.shape[1], 1)),-1)

        for i, layer in enumerate(self.nodePredict):
            h = layer(h)
            if i < (len(self.nodePredict)-1):
                h = activation(h)

        return  h.reshape(h.shape[:-1])


class smartFNN(torch.nn.Module):
    def __init__(self, inputSize, graph_latentLayer=[10048],FCLayer=[4048, 2048] ,  nodeTransferLayer = [4048, 1024, 1]):
        super(smartFNN, self).__init__()
        self.graphTrns = Graph_mlp(inputSize,  graph_latentLayer)
        FCLayer =  [graph_latentLayer[-1]]+ FCLayer
        self.FC = torch.nn.ModuleList([torch.nn.Linear(FCLayer[i], FCLayer[i + 1]) for i in range(len(FCLayer) - 1)])

        nodeTransferLayer =  [2* inputSize+ FCLayer[-1]] + nodeTransferLayer
        self.graphTranferlayers = torch.nn.ModuleList([torch.nn.Linear(nodeTransferLayer[i], nodeTransferLayer[i + 1]) for i in range(len(nodeTransferLayer) - 1)])

    def forward(self, in_tensor,subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.01)):
        graphEm = self.graphTrns(in_tensor, activation)

        for layer in self.FC:
            graphEm = layer(graphEm)
            graphEm = activation(graphEm)

        #merging the embedding
        z = torch.cat((torch.unsqueeze(in_tensor, 1).repeat(1, in_tensor.shape[1], 1, 1), torch.unsqueeze(in_tensor, 2).repeat(1, 1, in_tensor.shape[1], 1)),-1)
        z = torch.cat((z, graphEm.reshape([z.shape[0],1,1,graphEm.shape[-1]]).repeat(1,z.shape[1],z.shape[1],1)),-1)

        for i, layer in enumerate(self.graphTranferlayers):
            z = layer(z)
            if i<(len(self.graphTranferlayers)-1):
                z = activation(z)
        return z.reshape(z.shape[:-1])

class NodeGraphTransformerDecoder(torch.nn.Module):
    def __init__(self,input,lambdaDim,node_num, layers= [4096,2048,  1024]):
        super(NodeGraphTransformerDecoder, self).__init__()

        self.node_num = node_num

        self.lamda = torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
        self.graphTrns = Graph_mlp(input, [ 4096])
        self.nodeTrnslayers = node_mlp(1024 + input, [1024, lambdaDim])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        self.reset_parameters()

    def forward(self, in_tensor, subgraphs_indexes=None, activation=torch.nn.LeakyReLU(0.01)):
        if subgraphs_indexes!=None:
            z = []
            for i in range(len(subgraphs_indexes)):
                z.append(in_tensor[i][subgraphs_indexes[i]])
            in_tensor = torch.stack(z)
            del z
        # h = (in_tensor)
        h = self.graphTrns(in_tensor, torch.relu)

        for i in range(len(self.layers)):
            h = self.layers[i](h)
            h = activation(h)

        h = torch.cat((in_tensor, torch.unsqueeze(h, 1).repeat(1, in_tensor.shape[1], 1)),2)
        h = self.nodeTrnslayers(h, activation, applyActOnTheLastLyr=False)

        adj_list = torch.matmul(torch.matmul(h, self.lamda), h.permute(0, 2, 1))
        # adj_list = torch.matmul(h,h.permute(0, 2, 1))
        return adj_list

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)

# class NodeGraphTransformerDecoder(torch.nn.Module):
#     def __init__(self,input,lambdaDim,node_num, layers= [4096,2048,  1024]):
#         super(NodeGraphTransformerDecoder, self).__init__()
#
#         self.node_num = node_num
#
#         self.lamda = torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
#         self.graphTrns = Graph_mlp(input, [ 4096])
#         self.nodeTrnslayers = node_mlp(1024 + input, [1024, lambdaDim])
#         self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
#         self.graphlambda = Graph_mlp(lambdaDim, [1028])
#         layers = [1028, lambdaDim**2]
#         self.lambdaLayer = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
#         self.reset_parameters()
#
#     def forward(self, in_tensor, subgraphs_indexes=None, activation=torch.nn.LeakyReLU(0.01)):
#         if subgraphs_indexes!=None:
#             z = []
#             for i in range(len(subgraphs_indexes)):
#                 z.append(in_tensor[i][subgraphs_indexes[i]])
#             in_tensor = torch.stack(z)
#             del z
#         # h = (in_tensor)
#         h = self.graphTrns(in_tensor, activation)
#
#         for i in range(len(self.layers)):
#             h = self.layers[i](h)
#             h = activation(h)
#
#         h = torch.cat((in_tensor, torch.unsqueeze(h, 1).repeat(1, in_tensor.shape[1], 1)),2)
#         h = self.nodeTrnslayers(h, activation, applyActOnTheLastLyr=False)
#
#         graphEm = self.graphlambda(h , activation)
#
#         for i, layer in enumerate(self.lambdaLayer):
#             graphEm = layer(graphEm)
#             if i !=(len(self.lambdaLayer)-1):
#                 graphEm = activation(graphEm)
#
#         adj_list = torch.matmul(torch.matmul(h, graphEm.reshape(-1, h.shape[-1], h.shape[-1])), h.permute(0, 2, 1))
#         # adj_list = torch.matmul(torch.matmul(h, self.lamda), h.permute(0, 2, 1))
#         # adj_list = torch.matmul(h,h.permute(0, 2, 1))
#         return adj_list
#
#     def reset_parameters(self):
#         self.lamda = torch.nn.init.xavier_uniform_(self.lamda)

class NodeGraphTransformerDecoder2(torch.nn.Module):
    def __init__(self,input,lambdaDim,node_num, layers= [1024 ]):
        super(NodeGraphTransformerDecoder2, self).__init__()
        self.lamda =torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
        self.node_num = node_num
        self.PreNodeTrns = node_mlp(input, [1024,input])
        # self.PreNodeTrns2 = node_mlp(input, [1024,input])
        self.graphTrns = Graph_mlp(input, [ 1024, 1024])
        self.nodeTrnslayers = node_mlp(1024+input, [1024,1024 ,lambdaDim])
        self.reset_parameters()
        self.z = torch.rand([1, node_num, node_num])
    def forward(self, in_tensor, subgraphs_indexes=None, activation=torch.nn.ReLU()):
        if subgraphs_indexes!=None:
            z = []
            for i in range(len(subgraphs_indexes)):
                z.append(in_tensor[i][subgraphs_indexes[i]])
            in_tensor = torch.stack(z)
            del z
        # h = (torch.eye(in_tensor.shape[-2]))
        # z = self.z.repeat(in_tensor.shape[0], 1, 1).to(in_tensor.device)
        z = in_tensor
        # z = self.PreNodeTrns2(in_tensor)

        in_tensor = self.PreNodeTrns(in_tensor, torch.relu)
        h = self.graphTrns(in_tensor, torch.relu)
        h = torch.cat((z, torch.unsqueeze(h, 1).repeat(1, in_tensor.shape[1], 1)),2)
        h = self.nodeTrnslayers(h, torch.relu)

        adj_list = torch.matmul(torch.matmul(h, self.lamda), h.permute(0, 2, 1))
        # adj_list = torch.matmul(h,h.permute(0, 2, 1))
        return adj_list

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)


class FC_InnerDOTdecoder(torch.nn.Module):
    def __init__(self,input,output,laten_size ,layer=[1024,1024,1024]):
        super(FC_InnerDOTdecoder,self).__init__()
        self.lamda = torch.nn.Parameter(torch.Tensor(laten_size, laten_size))
        layer = [input]+layer + [output]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layer[i],layer[i+1]) for i in range(len(layer)-1)])
        self.reset_parameters()
    # def forward(self,Z):
    #     shape = Z.shape
    #     z = Z.reshape(shape[0],-1)
    #     for i in range(len(self.layers)):
    #         z  = self.layers[i](z)
    #         z = torch.tanh(z)
    #     # Z = torch.sigmoid(Z)
    # return z.reshape(shape[0], shape[-2], shape[-2])
    def forward(self, in_tensor, subgraphs_indexes=None,  activation=torch.nn.LeakyReLU(0.01)):
        h = in_tensor.reshape(in_tensor.shape[0],-1)
        for i in range(len(self.layers)):
            # if self.norm_layers != None:
            #     if len(h.shape) == 2:
            #         h = self.norm_layers[i](h)
            #     else:
            #         shape = h.shape
            #         h = h.reshape(-1, h.shape[-1])
            #         h = self.norm_layers[i](h)
            #         h = h.reshape(shape)
            # h = self.dropout(h)
            h = self.layers[i](h)
            if ((i!=(len(self.layers)-1))):
                h = activation(h)
        h = h.reshape(in_tensor.shape[0], in_tensor.shape[1],-1)
        # return torch.matmul(torch.matmul(h,self.lamda), h.permute(0, 2, 1))
        if subgraphs_indexes==None:
            adj_list= torch.matmul(torch.matmul(h, self.lamda),h.permute(0,2,1))
            return adj_list
        else:
            adj_list = []
            for i in range(h.shape[0]):
                adj_list.append(torch.matmul(torch.matmul(h[i][subgraphs_indexes[i]], self.lamda),h[i][subgraphs_indexes[i]].permute( 1, 0)))
            return torch.stack(adj_list)
    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)
#============================================================================
class kernel(torch.nn.Module):
    """
     this class return a list of kernel ordered by keywords in kernel_type
    """
    def __init__(self, **ker):
        """
        :param ker:
        kernel_type; a list of string which determine needed kernels
        """
        super(kernel, self).__init__()
        self.kernel_type = ker.get("kernel_type")
        kernel_set = set(self.kernel_type)

        if "in_degree_dist" in kernel_set or "out_degree_dist" in kernel_set:
            self.degree_hist = Histogram(ker.get("degree_bin_width").to(device), ker.get("degree_bin_center").to(device))

        if "RPF" in kernel_set:
            self.num_of_steps = ker.get("step_num")
            self.hist = Histogram(ker.get("bin_width"), ker.get("bin_center"))

        if "trans_matrix" in kernel_set:
            self.num_of_steps = ker.get("step_num")



    def forward(self,adj):
        vec = self.kernel_function(adj)
        # return self.hist(vec)
        return vec

    def kernel_function(self, adj): # TODO: another var for keeping the number of moments
        # ToDo: here we assumed the matrix is symetrix(undirected) which might not
        vec = []  # feature vector
        for kernel in self.kernel_type:
            if "in_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    # degree = adj[i, subgraph_indexes[i]][:, subgraph_indexes[i]].sum(1).view(1, -1)
                    degree = adj[i].sum(1).view(1, -1)
                    degree_hit.append(self.degree_hist(degree.to(device)))
                vec.append(torch.cat(degree_hit))
            if "out_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    degree = adj[i].sum(0).view(1, -1)
                    degree_hit.append(self.degree_hist(degree))
                vec.append(torch.cat(degree_hit))
            if "RPF" == kernel:
                raise("should be changed") #ToDo: need to be fixed
                tr_p = self.S_step_trasition_probablity(adj, self.num_of_steps)
                for i in range(len(tr_p)):
                    vec.append(self.hist(torch.diag(tr_p[i])))

            if "trans_matrix" == kernel:
                vec.extend(self.S_step_trasition_probablity(adj, self.num_of_steps))
                # vec = torch.cat(vec,1)

            if "tri" == kernel:  # compare the nodes degree in the given order
                tri, square = self.tri_square_count(adj)
                vec.append(tri), vec.append(square)

        return vec

    def tri_square_count(self, adj):
        ind = torch.eye(adj[0].shape[0]).to(device)
        adj = adj - ind
        two__ = torch.matmul(adj, adj)
        tri_ = torch.matmul(two__, adj)
        squares = torch.matmul(two__, two__)
        return (torch.diagonal(tri_, dim1=1, dim2=2), torch.diagonal(squares, dim1=1, dim2=2))

    def S_step_trasition_probablity(self, adj, s=4, ):
        """
         this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
        :param s: maximum step; the function will return s-step matrixes from 1 to s in a list
        :param Adj: adjacency matrix of the graph
        :return: a list in whcih the i-th elemnt is the i step transition probablity
        """
        # mask = torch.zeros(adj.shape).to(device)

        p1 = adj.to(device)
        # p1 = p1 * mask
        # ind = torch.eye(adj[0].shape[0])
        # p1 = p1 - ind
        TP_list = []
        # to save memory Use ineficient loop
        if dataset=="large_grid":
            p = []
            for i in range(adj.shape[0]):
                p.append(p1[i] * (p1[i].sum(1).float().clamp(min=1) ** -1))
            p1 = torch.stack(p)
        else:
            p1 = p1*(p1.sum(2).float().clamp(min=1) ** -1).view(adj.shape[0],adj.shape[1], 1)

        # p1[p1!=p1] = 0
        # p1 = p1 * mask

        if s>0:
            # TP_list.append(torch.matmul(p1,p1))
            TP_list.append( p1)
        for i in range(s-1):
            TP_list.append(torch.matmul(p1, TP_list[-1] ))
        return TP_list

# class kernel(torch.nn.Module):
#     """
#      this class return a list of kernel ordered by keywords in kernel_type
#     """
#     def __init__(self, **ker):
#         """
#         :param ker:
#         kernel_type; a list of string which determine needed kernels
#         """
#         super(kernel, self).__init__()
#         self.kernel_type = ker.get("kernel_type")
#         kernel_set = set(self.kernel_type)
#
#         if "in_degree_dist" in kernel_set or "out_degree_dist" in kernel_set:
#             self.degree_hist = Histogram(ker.get("degree_bin_width").to(device), ker.get("degree_bin_center").to(device))
#
#         if "RPF" in kernel_set:
#             self.num_of_steps = ker.get("step_num")
#             self.hist = Histogram(ker.get("bin_width"), ker.get("bin_center"))
#
#         if "trans_matrix" in kernel_set:
#             self.num_of_steps = ker.get("step_num")
#
#
#
#     def forward(self,adj, num_nodes):
#         vec = self.kernel_function(adj, num_nodes)
#         # return self.hist(vec)
#         return vec
#
#     def kernel_function(self, adj, num_nodes): # TODO: another var for keeping the number of moments
#         # ToDo: here we assumed the matrix is symetrix(undirected) which might not
#         vec = []  # feature vector
#         for kernel in self.kernel_type:
#             if "in_degree_dist" == kernel:
#                 degree_hit = []
#                 for i in range(adj.shape[0]):
#                     degree = adj[i,:num_nodes[i],:num_nodes[i]].sum(1).view(1, num_nodes[i])
#                     degree_hit.append(self.degree_hist(degree.to(device)))
#                 vec.append(torch.cat(degree_hit))
#             if "out_degree_dist" == kernel:
#                 degree_hit = []
#                 for i in range(adj.shape[0]):
#                     degree = adj[i, :num_nodes[i], :num_nodes[i]].sum(0).view(1, num_nodes[i])
#                     degree_hit.append(self.degree_hist(degree))
#                 vec.append(torch.cat(degree_hit))
#             if "RPF" == kernel:
#                 raise("should be changed") #ToDo: need to be fixed
#                 tr_p = self.S_step_trasition_probablity(adj, num_nodes, self.num_of_steps)
#                 for i in range(len(tr_p)):
#                     vec.append(self.hist(torch.diag(tr_p[i])))
#
#             if "trans_matrix" == kernel:
#                 vec.extend(self.S_step_trasition_probablity(adj, num_nodes, self.num_of_steps))
#                 # vec = torch.cat(vec,1)
#
#             if "tri" == kernel:  # compare the nodes degree in the given order
#                 tri, square = self.tri_square_count(adj)
#                 vec.append(tri), vec.append(square)
#
#         return vec
#
#     def tri_square_count(self, adj):
#         ind = torch.eye(adj[0].shape[0]).to(device)
#         adj = adj - ind
#         two__ = torch.matmul(adj, adj)
#         tri_ = torch.matmul(two__, adj)
#         squares = torch.matmul(two__, two__)
#         return (torch.diagonal(tri_, dim1=1, dim2=2), torch.diagonal(squares, dim1=1, dim2=2))
#
#     def S_step_trasition_probablity(self, adj, num_node, s=4, ):
#         """
#          this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
#         :param s: maximum step; the function will return s-step matrixes from 1 to s in a list
#         :param Adj: adjacency matrix of the graph
#         :return: a list in whcih the i-th elemnt is the i step transition probablity
#         """
#         mask = torch.zeros(adj.shape).to(device)
#         for i in range(adj.shape[0]):
#             mask[i,:num_node[i],:num_node[i]] = 1
#
#         p1 = adj.to(device)
#         p1 = p1 * mask
#         # ind = torch.eye(adj[0].shape[0])
#         # p1 = p1 - ind
#         TP_list = []
#         # to save memory Use ineficient loop
#         if dataset=="large_grid":
#             p = []
#             for i in range(adj.shape[0]):
#                 p.append(p1[i] * (p1[i].sum(1).float().clamp(min=1) ** -1))
#             p1 = torch.stack(p)
#         else:
#             p1 = p1*(p1.sum(2).float().clamp(min=1) ** -1).view(adj.shape[0],adj.shape[1], 1)
#
#         # p1[p1!=p1] = 0
#         # p1 = p1 * mask
#
#         if s>0:
#             # TP_list.append(torch.matmul(p1,p1))
#             TP_list.append( p1)
#         for i in range(s-1):
#             TP_list.append(torch.matmul(p1, TP_list[-1] ))
#         return TP_list

def test_(number_of_samples, model ,graph_size,max_size, path_to_save_g, remove_self=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    k = 0
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, max_size, model.latent_dim]))
            z = torch.randn_like(z)
            start_time = time.time()
            if type(model.decode) == GRAPHITdecoder:
                pass
                # adj_logit = model.decode(z.float(), features)
            elif type(model.decode) == RNNDecoder:
                adj_logit = model.decode(z.to(device).float(), [g_size])
            elif type(model.decode) in (FCdecoder, FC_InnerDOTdecoder):
                g_size = max_size
                z = torch.tensor(numpy.random.normal(size=[1, max_size, model.latent_dim]))
                z = torch.randn_like(z)
                adj_logit = model.decode(z.to(device).float())
            else:
                adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            sample_graph = sample_graph[:g_size,:g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_matrix(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g+str(k) +str(g_size)+ str(j) + dataset
            k+=1
            # plot and save the generated graph
            # plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            plotter.plotG(G, "generated" + dataset, file_name=f_name+"_ConnectedComponnents")
    # ======================================================
    # save nx files
    nx_f_name = path_to_save_g + "_" + dataset + "_" + decoder_type + "_" + args.model + "_" + task
    with open(nx_f_name, 'wb') as f:
        pickle.dump(generated_graph_list, f)
    # # ======================================================
    return generated_graph_list




def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, num_nodes, alpha, reconstructed_adj_logit, pos_wight, norm, node_num ):

    loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight)

    norm =    mean.shape[0] * mean.shape[1] * mean.shape[2]
    kl = (1/norm)* -0.5 * torch.sum(1+2*log_std - mean.pow(2)-torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum()/float(reconstructed_adj.shape[0]*reconstructed_adj.shape[1]*reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []
    for i in range(len(target_kernel_val)):
        l = torch.nn.MSELoss()
        step_loss = l(reconstructed_kernel_val[i].float(), target_kernel_val[i].float())
        each_kernel_loss.append(step_loss.cpu().detach().numpy()*alpha[i])
        kernel_diff += l(reconstructed_kernel_val[i].float(), target_kernel_val[i].float())*alpha[i]
    each_kernel_loss.append(loss.cpu().detach().numpy()*alpha[-2])
    each_kernel_loss.append(kl.cpu().detach().numpy()*alpha[-1])
    kernel_diff += loss*alpha[-2]
    kernel_diff += kl * alpha[-1]
    return kl , loss, acc, kernel_diff, each_kernel_loss

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

# test_(5, "results/multiple graph/cora/model" , [x**2 for x in range(5,10)])


# load the data
list_adj, list_x = list_graph_loader( dataset)
# list_adj, _ = permute(list_adj, None)
self_for_none = True
if (decoder_type)in  ("FCdecoder"):#,"FC_InnerDOTdecoder"
    self_for_none = True

if len(list_adj)==1:
    test_list_adj=list_adj.copy()
    list_graphs = Datasets(list_adj, self_for_none, None, None)
else:
    if task == "linkPrediction":
        max_size = max([x.shape[0] for x in list_adj])
    else:
        max_size = None
    list_adj, test_list_adj = data_split(list_adj)
    list_graphs = Datasets(list_adj, self_for_none, None,Max_num=max_size)

SubGraphNodeNum = subgraphSize if subgraphSize!=None else list_graphs.max_num_nodes
in_feature_dim = list_graphs.feature_size # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes


degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum, 1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

kernel_model = kernel(kernel_type = kernl_type, step_num = step_num,
                      bin_width= bin_width, bin_center=bin_center, degree_bin_center=degree_center, degree_bin_width=degree_width)

kernel_model_eval = kernel(kernel_type = kernl_type_eval, step_num = step_num_eval,
                           bin_width= bin_width, bin_center=bin_center, degree_bin_center=degree_center, degree_bin_width=degree_width)
# 225#

if decoder_type=="SBMdecoder":
    decoder = SBMdecoder_(hidden_2, lambda_dim = 256)
elif decoder_type=="SmarterSBMdecoder_":
    decoder = SmarterSBMdecoder_(hidden_2)
elif decoder_type =="smartFNN":
    decoder=smartFNN(hidden_2)
elif decoder_type == "NodeGraphTransformerDecoder":
    decoder = NodeGraphTransformerDecoder(hidden_2, 256, nodeNum)
elif decoder_type == "NodeGraphTransformerDecoder2":
    decoder = NodeGraphTransformerDecoder2(hidden_2, 256, nodeNum)
elif decoder_type == "GraphTransformerDecoder":
    # decoder = GraphTransformerDecoder(hidden_2, 256,subgraphSize if subgraphSize!=None else  nodeNum )
    decoder = GraphTransformerDecoder(hidden_2, 256,nodeNum )
elif decoder_type == "GraphTransformerDecoder2":
    # decoder = GraphTransformerDecoder(hidden_2, 256,subgraphSize if subgraphSize!=None else  nodeNum )
    decoder = GraphTransformerDecoder2(hidden_2, 256, nodeNum)
elif decoder_type=="FCdecoder":
    decoder= FCdecoder(list_graphs.max_num_nodes*hidden_2,list_graphs.max_num_nodes**2)
elif decoder_type == "FC_InnerDOTdecoder":
    # decoder = FC_InnerDOTdecoder(list_graphs.max_num_nodes * hidden_2, list_graphs.max_num_nodes *hidden_2,hidden_2, laten_size = 256)
    decoder =  FC_InnerDOTdecoder(list_graphs.max_num_nodes * hidden_2, list_graphs.max_num_nodes *256, laten_size = 256)
elif decoder_type=="GRAPHITdecoder":
    decoder = graphitDecoder(list_graphs.processed_Xs[0].shape[-1], hidden_2)
    # decoder = GRAPHITdecoder(hidden_2,25)
elif decoder_type=="GRAPHdecoder":
    decoder = GRAPHdecoder(hidden_2)
elif decoder_type=="GRAPHdecoder2":
    decoder = GRAPHdecoder(hidden_2,type="nn",)
elif decoder_type=="RNNDecoder":
    decoder = RNNDecoder(hidden_2)

model = kernelGVAE(in_feature_dim, hidden_1,  hidden_2,  kernel_model,decoder, [hidden_2]) # parameter namimng, it should be dimentionality of distriburion
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)


# pos_wight = torch.true_divide((list_graphs.max_num_nodes**2*len(list_graphs.processed_adjs)-list_graphs.toatl_num_of_edges),
#                               list_graphs.toatl_num_of_edges) # addrressing imbalance data problem: ratio between positve to negative instance
# pos_wight = torch.tensor(40.0)
# pos_wight/=10
num_nodes = list_graphs.max_num_nodes
#ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)

list_graphs.shuffle()
start = timeit.default_timer()
# Parameters
step =0
swith = False

min_loss = float('inf')
for epoch in range(epoch_number):
    list_graphs.shuffle()
    batch = 0
    for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
        from_ = iter
        to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+1)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)
        org_adj,x_s, node_num, subgraphs_indexes = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)
        if(type(decoder))in (GraphTransformerDecoder2, FCdecoder, FC_InnerDOTdecoder,  NodeGraphTransformerDecoder, GraphTransformerDecoder,SmarterSBMdecoder_, smartFNN): #
            node_num = len(node_num)*[list_graphs.max_num_nodes]
        # org_adj = torch.cat(org_adj).to(device)
        x_s = torch.cat(x_s)
        x_s = x_s.reshape(-1, x_s.shape[-1])

        model.train()


        subgraphs =[]
        for i  in range(len(org_adj)):
            subGraph = org_adj[i]
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i],:]
            # Converting sparse matrix to sparse tensor
            subGraph = torch.tensor(subGraph.todense())
            subgraphs.append(subGraph)

        subgraphs = torch.stack(subgraphs).to(device)
        pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())
        # pos_wight = torch.tensor(1)
        target_kelrnel_val = kernel_model(subgraphs)
        # target_kelrnel_val = kernel_model(org_adj, node_num)

        # batchSize = [org_adj.shape[0], org_adj.shape[1]]

        batchSize = [len(org_adj), org_adj[0].shape[0]]

        # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
        [graph.setdiag(1) for graph in org_adj]
        org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
        org_adj_dgl = dgl.batch(org_adj_dgl).to(device)


        reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val,reconstructed_adj_logit = model(org_adj_dgl.to(device), x_s.to(device), node_num, batchSize, subgraphs_indexes)
        kl_loss, reconstruction_loss, acc, kernel_cost,each_kernel_loss = OptimizerVAE(reconstructed_adj, generated_kernel_val, subgraphs, target_kelrnel_val, post_log_std, post_mean, num_nodes, alpha,reconstructed_adj_logit, pos_wight, 2,node_num)


        loss = kernel_cost


        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), None, *each_kernel_loss],tmp, redraw= redraw)  # ["Accuracy", "loss", "AUC"])

        step+=1
        optimizer.zero_grad()
        loss.backward()

        if keepThebest and min_loss>loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()

        if (epoch+batch) % visulizer_step == 0:
            pltr.redraw()
            if   dataset in synthesis_graphs:
                dir_generated_in_train = "generated_graph_train/"
                if not os.path.isdir(dir_generated_in_train):
                    os.makedirs(dir_generated_in_train)
                rnd_indx = random.randint(0,len(node_num)-1)
                sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
                sample_graph = sample_graph[:node_num[rnd_indx],:node_num[rnd_indx]]
                sample_graph[sample_graph >= 0.5] = 1
                sample_graph[sample_graph < 0.5] = 0
                G = nx.from_numpy_matrix(sample_graph)
                plotter.plotG(G, "generated" + dataset, file_name=dir_generated_in_train+str(epoch)+"grid")
                # if reconstruction_loss.item()<0.051276 and not swith:
                #     alpha[-1] *=2
                #     swith = True
        k_loss_str=""
        for indx,l in enumerate(each_kernel_loss):
            k_loss_str+=functions[indx+3]+":"
            k_loss_str+=str(l)+".   "

        print("Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
            epoch + 1,batch,  loss.item(), reconstruction_loss.item(), kl_loss.item(), acc),k_loss_str)

        batch+=1
print("trainning elbo:)")



# save the train loss for comparing the convergence
import json

file_name = decoder_type+"_"+dataset+"_"+args.model+"_elbo_loss.txt"

with open(file_name, "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2])+np.array(pltr.values_train[-1])), fp)

file_name = decoder_type+"_"+dataset+"_"+args.model+"_CrossEntropyLoss.txt"
with open(file_name, "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2])), fp)

file_name = decoder_type+"_"+dataset+"_"+args.model+"_train_loss.txt"
with open(file_name, "w") as fp:
    json.dump(pltr.values_train[1], fp)

for indx, l in enumerate(each_kernel_loss):
    k_loss_str += functions_eval[indx + 3] + ":"
    k_loss_str += str(l) + ".   "

# target_kelrnel_val = kernel_model_eval(org_adj)
# reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
#     org_adj.to(device), x_s.to(device), batchSize)
# rec_kelrnel_val = kernel_model_eval(reconstructed_adj)
# kl_loss, reconstruction_loss, acc, kernel_cost, each_kernel_loss = OptimizerVAE(reconstructed_adj, rec_kelrnel_val,
#                                                                                 org_adj, target_kelrnel_val,
#                                                                                 post_log_std, post_mean, num_nodes,
#                                                                                 alpha_eval, reconstructed_adj_logit,
#                                                                                 pos_wight, 2, node_num)
# print("Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
#         epoch + 1, batch, kernel_cost.item(), reconstruction_loss.item(), kl_loss.item(), acc), k_loss_str)
def RMlink(gr_adj, rate=.05):
    graph_adj = gr_adj.copy()
    index = []
    while len(index) < rate*len(graph_adj.data):
        index.append(random.randint(0,len(graph_adj.data)-1))
        index = list(np.unique(index))
    # index =  [random.randint(0,len(graph_adj.data)-1) for i in range(math.floor(rate*len(graph_adj.data)))]
    pos_list = [[graph_adj.nonzero()[0][i], graph_adj.nonzero()[1][i]] for i in index]

    ng_index = []
    while len(ng_index) < math.floor(rate*len(graph_adj.data)):
        i= random.randint(0,graph_adj.shape[0]-1)
        j = random.randint(0,graph_adj.shape[0]-1)
        if graph_adj[i,j] ==0:
            ng_index.append([i,j])
    graph_adj.data[index] = 0
    scipy.sparse.csr_matrix.eliminate_zeros(graph_adj)
    return graph_adj, pos_list, ng_index

#save the log plot on the current directory
from pathlib import Path
Path(graph_save_path).mkdir(parents=True, exist_ok=True)
pltr.save_plot(graph_save_path+"KernelVGAE_log_plot")
if task == "linkPrediction":
    np.random.seed(0)
    random.seed(0)
    masked_graphs = []
    pos_edges_list = []
    Ng_edges_list = []
    for i, graph in enumerate(test_list_adj):
        graph, pos_edges, neg_edges = RMlink(graph)
        masked_graphs.append(graph)
        Ng_edges_list.append(neg_edges)
        pos_edges_list.append(pos_edges)
    test_list_graphs = Datasets(masked_graphs, self_for_none, None, True, max_size)

    adj, x_s, node_num = test_list_graphs.get__(0, len(test_list_adj), self_for_none)
    adj = torch.cat(adj).to(device)
    x_s = torch.cat(x_s).to(device)
    # forwarding graphs with removed linked
    reconstrted_graphs,*_ = model(adj, x_s, node_num)

    # model Evaluation
    print("=====================================")
    print("Result on Link Prediction Task")

    auc, val_acc, val_ap, conf_mtrx = roc_auc_estimator_onGraphList(pos_edges_list, Ng_edges_list, reconstrted_graphs.cpu().detach().numpy(),
                                                                    test_list_adj)
    print("Test_acc: {:03f}".format(val_acc), " | Test_auc: {:03f}".format(auc), " | Test_AP: {:03f}".format(val_ap))
    print("Confusion matrix: \n", conf_mtrx)




stop = timeit.default_timer()
print("trainning time:", str(stop-start))
# torch.save(model, PATH)
if dataset=="DD" or  dataset=="ogbg-molbbbp" or dataset=="IMDbMulti":
    smp_num =1
else: smp_num = 10
generated_graphs = test_(smp_num, model , [x.shape[0] for x in test_list_adj],list_graphs.max_num_nodes, graph_save_path)


from stat_rnn import mmd_eval
if dataset in synthesis_graphs:
    mmd_eval(generated_graphs,[nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
    print("====================================================")
    if   keepThebest != False:
        print("for the model with the minimum loss" + str(min_loss) + " we have:")
        model.load_state_dict(torch.load("model"))
        generated_graphs = test_(smp_num, model, [x.shape[0] for x in test_list_adj], list_graphs.max_num_nodes,
                                 graph_save_path)
        mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
    # print("result for subgraph with maximum connected componnent")
    # generated_graphs = [G.subgraph(max(nx.connected_components(G), key=len)) for G in generated_graphs]
    # mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
    # print("====================================================")
    # print("comparing train and test")
    # G_train = [nx.from_numpy_matrix(graph.toarray()) for graph in list_adj]
    # for G in G_train:
    #     G.remove_edges_from(nx.selfloop_edges(G))
    # mmd_eval(G_train, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])

else:
    mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])

exit(0)




target_kelrnel_val = kernel_model_eval(org_adj, node_num)
rec_kelrnel_val = kernel_model_eval(reconstructed_adj, node_num)
kl_loss, reconstruction_loss, acc, kernel_cost, each_kernel_loss = OptimizerVAE(reconstructed_adj, rec_kelrnel_val,
                                                                                org_adj, target_kelrnel_val,
                                                                                post_log_std, post_mean, num_nodes,
                                                                                alpha_eval, reconstructed_adj_logit,
                                                                                pos_wight, 2, node_num)

# save the train loss for comparing the convergence
import json
file_name = dataset+"_"+args.model+"_elbo_loss.txt"
with open(file_name, "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2])+np.array(pltr.values_train[-1])), fp)

file_name = dataset+"_"+args.model+"_CrossEntropyLoss.txt"
with open(file_name, "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2])), fp)

file_name = dataset+"_"+args.model+"_train_loss.txt"
with open(file_name, "w") as fp:
    json.dump(pltr.values_train[1], fp)

for indx, l in enumerate(each_kernel_loss):
    k_loss_str += functions_eval[indx + 3] + ":"
    k_loss_str += str(l) + ".   "
print("Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
    epoch + 1, batch, kernel_cost.item(), reconstruction_loss.item(), kl_loss.item(), acc), k_loss_str)
def RMlink(gr_adj, rate=.05):
    graph_adj = gr_adj.copy()
    index = []
    while len(index) < rate*len(graph_adj.data):
        index.append(random.randint(0,len(graph_adj.data)-1))
        index = list(np.unique(index))
    # index =  [random.randint(0,len(graph_adj.data)-1) for i in range(math.floor(rate*len(graph_adj.data)))]
    pos_list = [[graph_adj.nonzero()[0][i], graph_adj.nonzero()[1][i]] for i in index]

    ng_index = []
    while len(ng_index) < math.floor(rate*len(graph_adj.data)):
        i= random.randint(0,graph_adj.shape[0]-1)
        j = random.randint(0,graph_adj.shape[0]-1)
        if graph_adj[i,j] ==0:
            ng_index.append([i,j])
    graph_adj.data[index] = 0
    scipy.sparse.csr_matrix.eliminate_zeros(graph_adj)
    return graph_adj, pos_list, ng_index

#save the log plot on the current directory
pltr.save_plot(graph_save_path+"KernelVGAE_log_plot")
if task == "linkPrediction":
    np.random.seed(0)
    random.seed(0)
    masked_graphs = []
    pos_edges_list = []
    Ng_edges_list = []
    for i, graph in enumerate(test_list_adj):
        graph, pos_edges, neg_edges = RMlink(graph)
        masked_graphs.append(graph)
        Ng_edges_list.append(neg_edges)
        pos_edges_list.append(pos_edges)
    test_list_graphs = Datasets(masked_graphs, self_for_none, None, True, max_size)

    adj, x_s, node_num = test_list_graphs.get__(0, len(test_list_adj), self_for_none)
    adj = torch.cat(adj).to(device)
    x_s = torch.cat(x_s).to(device)
    # forwarding graphs with removed linked
    reconstrted_graphs,*_ = model(adj, x_s, node_num)

    # model Evaluation
    print("=====================================")
    print("Result on Link Prediction Task")

    auc, val_acc, val_ap, conf_mtrx = roc_auc_estimator_onGraphList(pos_edges_list, Ng_edges_list, reconstrted_graphs.cpu().detach().numpy(),
                                                                    test_list_adj)
    print("Test_acc: {:03f}".format(val_acc), " | Test_auc: {:03f}".format(auc), " | Test_AP: {:03f}".format(val_ap))
    print("Confusion matrix: \n", conf_mtrx)




stop = timeit.default_timer()
print("trainning time:", str(stop-start))
# torch.save(model, PATH)
if dataset=="DD" or  dataset=="ogbg-molbbbp":
    smp_num =1
else: smp_num = 10
generated_graphs = test_(smp_num, model , [x.shape[0] for x in test_list_adj],list_graphs.max_num_nodes, graph_save_path)


from stat_rnn import mmd_eval
if dataset in synthesis_graphs:
    mmd_eval(generated_graphs,[nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
    print("====================================================")
    print("result for subgraph with maximum connected componnent")
    generated_graphs = [G.subgraph(max(nx.connected_components(G), key=len)) for G in generated_graphs]
    mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
    print("====================================================")
    print("comparing train and test")
    G_train = [nx.from_numpy_matrix(graph.toarray()) for graph in list_adj]
    for G in G_train:
        G.remove_edges_from(nx.selfloop_edges(G))
    mmd_eval(G_train, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])

else:
    mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])


# subgraphSize = None
# parser = argparse.ArgumentParser(description='Kernel VGAE')
# parser.add_argument('-e', dest="epoch_number" , default=20000, help="Number of Epochs")
# parser.add_argument('-v', dest="Vis_step", default=400, help="model learning rate")
#
# parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
# parser.add_argument('-lr', dest="lr", default=0.0001, help="model learning rate") # for RNN decoder use 0.0001
# parser.add_argument('-NSR', dest="negative_sampling_rate", default=1, help="the rate of negative samples which shold be used in each epoch; by default negative sampling wont use")
# parser.add_argument('-dataset', dest="dataset", default="lobster", help="possible choices are:  ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")#citeceer: ego; DD:protein
#
# parser.add_argument('-NofCom', dest="num_of_comunities", default=256, help="Number of comunites")
# parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
# parser.add_argument('-graph_save_path', dest="graph_save_path", default="develop_lobster_kernel/", help="the direc to save generated synthatic graphs")
# parser.add_argument('-f', dest="use_feature" , default=True, help="either use features or identity matrix")
# parser.add_argument('-Split', dest="split_the_data_to_train_test" , default=True, help="either use features or identity matrix; for synthasis data default is False")
# parser.add_argument('-PATH', dest="PATH" , default="model", help="a string which determine the path in wich model will be saved")
# parser.add_argument('-decoder', dest="decoder" , default="GraphTransformerDecoder", help="the decoder type,GraphTransformerDecoder, SBMdecoder, FC_InnerDOTdecoder, GRAPHdecoder,FCdecoder,")
# parser.add_argument('-batchSize', dest="batchSize" , default=100, help="the size of each batch")
# parser.add_argument('-UseGPU', dest="UseGPU" , default=True, help="either use GPU or not if availabel")
# parser.add_argument('-model', dest="model" , default="kernel", help="sbm, kipf or kernel")
# parser.add_argument('-device', dest="device" , default="cuda:0", help="Which device should be used")
# parser.add_argument('-task', dest="task" , default="graphGeneration", help="linkPrediction or graphGeneration")

# class GraphTransformerDecoder(torch.nn.Module):
#     def __init__(self,input,lambdaDim,SubGraphNodeNum, layers= [1024 ]):
#         super(GraphTransformerDecoder, self).__init__()
#         self.lamda =torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
#         self.SubGraphNodeNum = SubGraphNodeNum
#         layers = [512]  +layers+ [lambdaDim*SubGraphNodeNum]
#         self.graphTrns = Graph_mlp(input, [512])
#         self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
#         self.reset_parameters()
#
#     def forward(self, in_tensor, subgraphs_indexes=None, activation=torch.nn.ReLU()):
#         # if subgraphs_indexes!=None:
#         #     z = []
#         #     for i in range(len(subgraphs_indexes)):
#         #         z.append(in_tensor[i][subgraphs_indexes[i]])
#         #     in_tensor = torch.stack(z)
#         #     del z
#
#         Graph_num = in_tensor.shape[0]
#         h = self.graphTrns(in_tensor, torch.relu)
#         del in_tensor
#         for i in range(len(self.layers)):
#             h = self.layers[i](h)
#             h = activation(h)
#         h = h.reshape(Graph_num, self.SubGraphNodeNum, -1)
#
#         if subgraphs_indexes==None:
#             adj_list= torch.matmul(torch.matmul(h, self.lamda),h.permute(0,2,1))
#             return adj_list
#         else:
#             adj_list = []
#             for i in range(h.shape[0]):
#                 adj_list.append(torch.matmul(torch.matmul(h[i][subgraphs_indexes[i]], self.lamda),h[i][subgraphs_indexes[i]].permute( 1, 0)))
#             return torch.stack(adj_list)
#
#     def reset_parameters(self):
#         self.lamda = torch.nn.init.xavier_uniform_(self.lamda)

