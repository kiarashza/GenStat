# Unexpected error: <class 'OSError'>
# Unexpected error: <class 'OSError'>
# Unexpected error: <class 'OSError'>
# Unexpected error: <class 'ZeroDivisionError'>
# degree 0.07535966601684341 clustering 0.048087770412737596 orbits -1
import logging
from stat_rnn import mmd_eval, Diam_stats
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
import Aggregation
import random as random
import time
import timeit
import dgl
from stat_rnn import mmd_eval
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



subgraphSize = None
keepThebest = False

parser = argparse.ArgumentParser(description='Kernel VGAE')


parser.add_argument('-e', dest="epoch_number" , default=20000, help="Number of Epochs", type=int)
parser.add_argument('-v', dest="Vis_step", default=2000, help="model learning rate")
# parser.add_argument('-AE', dest="AutoEncoder", default=True, help="either update the log plot each step")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.0003, help="model learning rate") # for RNN decoder use 0.0001
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1, help="the rate of negative samples which shold be used in each epoch; by default negative sampling wont use")
parser.add_argument('-dataset', dest="dataset", default="IMDBBINARY", help="possible choices are:   IMDbMulti,wheel_graph, star, triangular_grid, multi_community, NCI1, ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")#citeceer: ego; DD:protein
parser.add_argument('-NofCom', dest="num_of_comunities", default=256, help="Number of comunites")
parser.add_argument('-graphEmDim', dest="graphEmDim", default=1024, help="the simention of graphEmbedingLAyer")
parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
parser.add_argument('-graph_save_path', dest="graph_save_path", default=None, help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature" , default=True, help="either use features or identity matrix")
parser.add_argument('-Split', dest="split_the_data_to_train_test" , default=True, help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-PATH', dest="PATH" , default="model", help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder" , default="FC", help="the decoder type, FC or SBM")
parser.add_argument('-encoder', dest="encoder_type" , default="AvePool", help="the encoder")    #"diffPool" "AvePool" "GCNPool"
parser.add_argument('-batchSize', dest="batchSize" , default=64, help="the size of each batch")
parser.add_argument('-UseGPU', dest="UseGPU" , default=True, help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model" , default="KernelAugmentedWithTotalNumberOfTriangles", help="TotalNumberOfTriangles ,KernelAugmentedWithTotalNumberOfTriangles, kipf or kernel")
parser.add_argument('-device', dest="device" , default="cuda:0", help="Which device should be used")
parser.add_argument('-task', dest="task" , default="graphGeneration", help="linkPrediction, GraphCompletion, graphClasssification or graphGeneration")
parser.add_argument('-BFS', dest="bfsOrdering" , default=True, help="use bfs for graph permutations", type=bool)
parser.add_argument('-directed', dest="directed" , default=False, help="is the dataset directed?!", type=bool)
parser.add_argument('-beta', dest="beta" , default=None, help="beta coefiicieny", type=float)
parser.add_argument('-limited_to', dest="limited_to" , default=60, help="How many instance you want to pick, its for reducing trainning time in dev", type=float)
parser.add_argument('-ideal_Evalaution', dest="ideal_Evalaution" , default=False, help="if you want to comapre the 50%50 subset of dataset comparision?!", type=bool)


args = parser.parse_args()
# torch.autograd.set_detect_anomaly(True)




encoder_type = args.encoder_type
ideal_Evalaution = args.ideal_Evalaution
graphEmDim = args.graphEmDim
visulizer_step = args.Vis_step
redraw = args.redraw
device = args.device
task = args.task
limited_to = args.limited_to
directed = args.directed
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
graph_save_path = args.graph_save_path
split_the_data_to_train_test = args.split_the_data_to_train_test


if graph_save_path==None:
    graph_save_path = "MMD_"+encoder_type+"_"+decoder_type+"_"+dataset+"_"+task+"_"+args.model+"BFS"+str(args.bfsOrdering)+str(args.epoch_number) +str(time.time())+"/"
from pathlib import Path
Path(graph_save_path).mkdir(parents=True, exist_ok=True)


# maybe to the beest way
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=graph_save_path+'log.log', filemode='w',  level=logging.INFO)





# **********************************************************************
# setting
print("KernelVGAE SETING: "+str(args))
logging.info("KernelVGAE SETING: "+str(args))
PATH = args.PATH # the dir to save the with the best performance on validation data


kernl_type = []

if args.model == "ThreeStepPath":
    kernl_type = ["ThreeStepPath"]

if args.model == "TrianglesOfEachNode":
    kernl_type = ["TrianglesOfEachNode"]


if args.model == "KernelAugmentedWithTotalNumberOfTriangles":
    kernl_type = [ "clusteringCoefficient","HistogramOfRandomWalks","in_degree_dist", "out_degree_dist", "TotalNumberOfTriangles","ReachabilityInKsteps","HistogramOfCycles"] # "clusteringCoefficient", "cycles_number","ReachabilityInKsteps" ReachabilityInKsteps "steps_of_reachability"
    max_size_of_cyc = 2
    steps_of_reachability = 2
    step_num = 2

    # max_size_of_cyc = 8
    # steps_of_reachability = 5
    # step_num = 5

    if  dataset == "large_grid":

        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 100]
    elif dataset == "small_triangular_grid":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 200]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 1500]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    elif dataset == "ogbg-molbbbp":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 200]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 1500]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    elif dataset == "PTC":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    elif dataset == "PVGAErandomGraphs":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    elif dataset == "FIRSTMM_DB":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 100]
    elif dataset == "DD":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 200]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 1000]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 30, 600]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 200]
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 1200]
    elif dataset=="grid" or dataset == "small_grid":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 10]
    elif dataset=="lobster" or dataset == "IMDbMulti" or dataset == "IMDBBINARY":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        alpha = [ 1, 1, 1, 1,1,1,1,1,0,1,1,1, 1]
    elif dataset == "wheel_graph":

        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 500000, 200000000]
    elif dataset == "triangular_grid":
        alpha = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    elif dataset == "tree":
        alpha = [1, 1, 1, 1, 1, 1, 1,1, 1000000, 10000000]
elif args.model == "kipf":
    alpha= [ 1,1]
    step_num = 0


AutoEncoder = False
if task=="graphClasssification":
    AutoEncoder = True

if AutoEncoder == True:
    alpha[-1] = 0

if task=="GraphCompletion":
    alpha[-1]=0

if args.beta!=None:
    alpha[-1] = args.beta

print("kernl_type:"+str(kernl_type))
print("alpha: "+str(alpha) +" num_step:"+str(step_num))


logging.info("kernl_type:"+str(kernl_type))
logging.info("alpha: "+str(alpha) +" num_step:"+str(step_num))


bin_center = torch.tensor([[x / 10000] for x in range(0, 1000, 1)])
bin_width = torch.tensor([[9000] for x in range(0, 1000, 1)])# with is propertion to revese of this value;

device = torch.device(device if torch.cuda.is_available() and use_gpu else "cpu")
print("the selected device is :", device)
logging.info("the selected device is :"+ str(device))

#========================================================================
# setting the plots legend
functions= ["Accuracy", "loss"]
if args.model == "kernel" or args.model =="KernelAugmentedWithTotalNumberOfTriangles":

    # functions.extend(["Kernel"+str(i) for i in range(step_num)])
    for ker in kernl_type:
        if ker == "HistogramOfCycles":
            functions.extend(["HistogramOfCycles_" + str(i+3) for i in range(max_size_of_cyc-2)])
            # functions.extend(["NumbetOfCycles_" + str(i + 3) for i in range(max_size_of_cyc - 2)])
        elif ker == "HistogramOfRandomWalks":
            functions.extend(["RandWalksHist" + str(i) for i in range(step_num)])
            # functions.extend(["NumberOfRandWalks_" + str(i) for i in range(step_num)])
        elif ker == "cycles_number":
            functions.extend(["CyclHist" + str(i) for i in range(step_num)])
        elif ker == "ReachabilityInKsteps":
            functions.extend(["ReachabilityInKsteps" + str(i) for i in range(steps_of_reachability)])
        else:
            functions.append(ker)







functions.append("KL-D")

#========================================================================



pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log",functions=functions)


split_the_data_to_train_test = False


# **********************************************************************
#Encoders
# class GCNEncoder(torch.nn.Module):
#     def __init__(self, in_feature_dim, hidden, GraphLatntDim):
#         super(GCNEncoder, self).__init__()
#
#         self.firstnorm_layer = torch.nn.LayerNorm(hidden,elementwise_affine=False)
#         self.third_norm_layer = torch.nn.LayerNorm(GraphLatntDim, elementwise_affine=False)
#         self.Second_norm_layer = torch.nn.LayerNorm(hidden, elementwise_affine=False)
#
#         self.first_conv_layer = dgl.nn.pytorch.conv.GraphConv(in_feature_dim, hidden, activation=None, bias=True,weight=True)
#         self.second_conv_layer = dgl.nn.pytorch.conv.GraphConv(hidden, hidden, activation=None, bias=True, weight=True)
#         self.third_conv_layer = dgl.nn.pytorch.conv.GraphConv(hidden, hidden, activation=None, bias=True, weight=True)
#         self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
#         self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
#         self.poolingLayer = Aggregation.GcnPool(hidden, GraphLatntDim)
#
#     def forward(self, graph, features, batchSize, activation= torch.nn.LeakyReLU(0.01)):
#         h = self.first_conv_layer(graph, features)
#         h= activation(h)
#         h = self.firstnorm_layer(h)
#
#         h = self.second_conv_layer(graph, h)
#         h = self.Second_norm_layer(h)
#
#         h = self.third_conv_layer(graph, h)
#         h = activation(h)
#         h = activation(h).reshape(*batchSize, -1)
#
#         h = self.poolingLayer(h, activation)
#
#         h = self.third_norm_layer(h)
#         mean = self.stochastic_mean_layer(h,activation = lambda x:x)
#         log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)
#
#         return  mean, log_std

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_feature_dim, hiddenLayers = [256, 256, 256], GraphLatntDim=1024):
        super(GCNEncoder, self).__init__()

        hiddenLayers = [in_feature_dim]+hiddenLayers
        self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(hiddenLayers[i+1],elementwise_affine=False) for i in range(len(hiddenLayers) - 1)])
        self.normLayers.append(torch.nn.LayerNorm(GraphLatntDim, elementwise_affine=False))
        self.GCNlayers = torch.nn.ModuleList([dgl.nn.pytorch.conv.GraphConv(hiddenLayers[i], hiddenLayers[i + 1], activation=None, bias=True,weight=True) for i in range(len(hiddenLayers) - 1)])


        self.poolingLayer = Aggregation.GcnPool(hiddenLayers[-1], GraphLatntDim)

        self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])


    def forward(self, graph, features, batchSize, activation= torch.nn.LeakyReLU(0.01)):
        h = features
        for i in range(len(self.GCNlayers)):
            h= self.GCNlayers[i](graph, h)
            h= activation(h)
            h = self.normLayers[i](h)


        h = h.reshape(*batchSize, -1)

        h = self.poolingLayer(h, activation)

        h = self.normLayers[-1](h)
        mean = self.stochastic_mean_layer(h,activation = lambda x:x)
        log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)

        return  mean, log_std


class FCEncoder(torch.nn.Module):
    def __init__(self, in_feature_dim, hiddenLayers = [256, 256, 256], GraphLatntDim=1024):
        super(AveEncoder, self).__init__()

        hiddenLayers = [in_feature_dim]+hiddenLayers + [GraphLatntDim]
        self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(hiddenLayers[i+1],elementwise_affine=False) for i in range(len(hiddenLayers) - 1)])

        self.GCNlayers = torch.nn.ModuleList([dgl.nn.pytorch.conv.GraphConv(hiddenLayers[i], hiddenLayers[i + 1], activation=None, bias=True,weight=True) for i in range(len(hiddenLayers) - 1)])


        self.poolingLayer = Aggregation.AvePool()

        self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])


    def forward(self, graph, features, batchSize, activation= torch.nn.LeakyReLU(0.01)):
        h = features
        for i in range(len(self.GCNlayers)):
            h= self.GCNlayers[i](graph, h)
            h= activation(h)
            if((i<len(self.GCNlayers)-1)):
                h = self.normLayers[i](h)


        h = h.reshape(*batchSize, -1)

        h = self.poolingLayer(h)

        h = self.normLayers[-1](h)
        mean = self.stochastic_mean_layer(h,activation = lambda x:x)
        log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)

        return  mean, log_std


class AveEncoder(torch.nn.Module):
    def __init__(self, in_feature_dim, hiddenLayers = [256, 256, 256], GraphLatntDim=1024):
        super(AveEncoder, self).__init__()

        hiddenLayers = [in_feature_dim]+hiddenLayers + [GraphLatntDim]
        self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(hiddenLayers[i+1],elementwise_affine=False) for i in range(len(hiddenLayers) - 1)])
        self.normLayers.append(torch.nn.LayerNorm(hiddenLayers[-1],elementwise_affine=False))
        self.GCNlayers = torch.nn.ModuleList([dgl.nn.pytorch.conv.GraphConv(hiddenLayers[i], hiddenLayers[i + 1], activation=None, bias=True,weight=True) for i in range(len(hiddenLayers) - 1)])


        self.poolingLayer = Aggregation.AvePool()

        self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])


    def forward(self, graph, features, batchSize, activation= torch.nn.LeakyReLU(0.01)):
        h = features
        for i in range(len(self.GCNlayers)):
            h= self.GCNlayers[i](graph, h)
            h= activation(h)
            # if((i<len(self.GCNlayers)-1)):
            h = self.normLayers[i](h)


        h = h.reshape(*batchSize, -1)

        h = self.poolingLayer(h)

        h = self.normLayers[-1](h)
        mean = self.stochastic_mean_layer(h,activation = lambda x:x)
        log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)

        return  mean, log_std


class diffPoolEncoder(torch.nn.Module):
    def __init__(self, in_feature_dim, GraphLatntDim=1024, PropagationFunction = "GraphSage"):
        super(diffPoolEncoder, self).__init__()
        import diffPool
        # self.graphSageLayer = diffPool.GraphSageLayer(in_feature_dim, GraphLatntDim, torch.nn.LeakyReLU(0.01), dropout = 0, aggregator_type = "mean")
        if PropagationFunction == "GraphSage":
            self.graphSageLayer = diffPool.GraphSage(in_feature_dim, GraphLatntDim, GraphLatntDim, 1, torch.nn.LeakyReLU(0.01),
                                                     dropout=0, aggregator_type="mean")
        else:
            self.graphSageLayer = diffPool.GraphGCN(in_feature_dim, GraphLatntDim, GraphLatntDim, 1, torch.nn.LeakyReLU(0.01),
                                                    dropout=0, aggregator_type="mean")

        self.poolingLayer = diffPool.DiffPoolBatchedGraphLayer(input_dim = GraphLatntDim, assign_dim = 1, output_feat_dim = GraphLatntDim,
                                                               activation = torch.nn.LeakyReLU(0.01), dropout = 0, aggregator_type=None, link_pred=None)

        self.normLayer = torch.nn.LayerNorm(GraphLatntDim,elementwise_affine=False)

        self.stochastic_mean_layer = node_mlp(GraphLatntDim, [GraphLatntDim])
        self.stochastic_log_std_layer = node_mlp(GraphLatntDim, [GraphLatntDim])


    def forward(self, graph, features, batchSize, activation= torch.nn.LeakyReLU(0.01)):
        h = self.graphSageLayer(graph, features)
        # h = h.reshape(*batchSize, -1)
        _, h = self.poolingLayer(graph, h)
        #
        h = self.normLayer(h)
        mean = self.stochastic_mean_layer(h,activation = lambda x:x)
        log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)

        return  mean, log_std

# **********************************************************************
class kernelGVAE(torch.nn.Module):
    def __init__(self,  ker,encoder, decoder, AutoEncoder, graphEmDim = 4096):
        super(kernelGVAE, self).__init__()
        self.embeding_dim = graphEmDim
        self.kernel = ker #TODO: bin and width whould be determined if kernel is his
        self.AutoEncoder = AutoEncoder
        self.decode = decoder
        self.encode = encoder

        self.stochastic_mean_layer = node_mlp(self.embeding_dim, [self.embeding_dim])
        self.stochastic_log_std_layer = node_mlp(self.embeding_dim, [self.embeding_dim])

    def forward(self, graph, features, num_node,batchSize , subgraphs_indexes):
        """
        :param graph: normalized adjacency matrix of graph
        :param features: normalized node feature matrix
        :return:
        """
        mean, log_std = self.encode( graph, features, batchSize)
        samples = self.reparameterize(mean, log_std, node_num)
        reconstructed_adj_logit = self.decode(samples, subgraphs_indexes)
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)

        kernel_value = self.kernel(reconstructed_adj)
        return reconstructed_adj, samples, mean, log_std, kernel_value, reconstructed_adj_logit

    def reparameterize(self, mean, log_std, node_num):
        if self.AutoEncoder == True:
            return mean
        var = torch.exp(log_std).pow(2)
        eps = torch.randn_like(var)
        sample = eps.mul(var).add(mean)

        return sample



class NodeUpsampling(torch.nn.Module):
    def __init__(self, InNode_num , outNode_num, InLatent_dim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num*InLatent_dim, InLatent_dim*outNode_num)

    def forward(self,  inTensor, activation= torch.nn.LeakyReLU(0.001)):
        Z  = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(Z.reshpe(inTensor.shape[0], -1).permute(0,2,1) ,inTensor)

        return activation(Z)

class LatentMtrixTransformer(torch.nn.Module):
    def __init__(self, InNode_num , InLatent_dim=None, OutLatentDim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num*InLatent_dim, InNode_num*OutLatentDim)

    def forward(self,  inTensor, activation= torch.nn.LeakyReLU(0.001)):
        Z  = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(inTensor, Z.reshpe(inTensor.shape[-1], -1)  )

        return activation(Z)
# class Histogram(torch.nn.Module):
#     # this is a soft histograam Function.
#     #for deails check section "3.2. The Learnable Histogram Layer" of
#     # "Learnable Histogram: Statistical Context Features for Deep Neural Networks"
#     def __init__(self, bin_width = None, bin_centers = None):
#         super(Histogram, self).__init__()
#         self.bin_width = bin_width.to(device)
#         self.bin_center = bin_centers.to(device)
#         if self.bin_width == None:
#             self.prism()
#         else:
#             self.bin_num = self.bin_width.shape[0]
#
#     def forward(self, vec):
#         #REceive a vector and return the soft histogram
#
#         #comparing each element with each of the bin center
#         score_vec = vec.view(vec.shape[0],1, vec.shape[1], ) - self.bin_center
#         # score_vec = vec-self.bin_center
#         score_vec = 1-torch.abs(score_vec)*self.bin_width
#         score_vec = torch.relu(score_vec)
#         return score_vec.sum(2)
#
#     def prism(self):
#         pass

class Histogram(torch.nn.Module):
    # this is a soft histograam Function.
    #for deails check section "3.2. The Learnable Histogram Layer" of
    # "Learnable Histogram: Statistical Context Features for Deep Neural Networks"
    def __init__(self, bin_width = None, bin_centers = None, normalizer = 1):
        super(Histogram, self).__init__()
        self.bin_width = bin_width.to(device)
        self.bin_center = bin_centers.to(device)
        self.norm = torch.tensor(normalizer).to(device)
        if self.bin_width == None:
            self.prism()
        else:
            self.bin_num = self.bin_width.shape[0]

    def forward(self, vec):
        #REceive a vector and return the soft histogram

        #comparing each element with each of the bin center
        score_vec = (vec.view(vec.shape[0],1, vec.shape[1], ) - self.bin_center)/self.norm
        # score_vec = vec-self.bin_center
        score_vec = 1-torch.abs(score_vec)*self.bin_width
        score_vec = torch.relu(score_vec)**2

        return score_vec.sum(2)

    def prism(self):
        pass
class GraphTransformerDecoder(torch.nn.Module):
    def __init__(self,input,lambdaDim,SubGraphNodeNum, layers= [128 ]):
        super(GraphTransformerDecoder, self).__init__()
        self.lamda =torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
        self.SubGraphNodeNum = SubGraphNodeNum
        layers = [input] + [2048, 1024]+[lambdaDim*SubGraphNodeNum]
        self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i],elementwise_affine=False) for i in range(len(layers) - 1)])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.reset_parameters()

    def forward(self, in_tensor, subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.001)):
        for i in range(len(self.layers)):
            # in_tensor = self.normLayers[i](in_tensor)
            in_tensor = self.layers[i](in_tensor)
            if i !=len((self.layers))-1:
                in_tensor = activation(in_tensor)
                in_tensor = self.normLayers[i](in_tensor)
        in_tensor = in_tensor.reshape(in_tensor.shape[0], self.SubGraphNodeNum, -1)

        # adj_list= torch.matmul(torch.matmul(in_tensor, self.lamda),in_tensor.permute(0,2,1))
        # return adj_list
        if subgraphs_indexes==None:
            adj_list= torch.matmul(torch.matmul(in_tensor, self.lamda),in_tensor.permute(0,2,1))
            return adj_list
        else:
            adj_list = []
            for i in range(in_tensor.shape[0]):
                adj_list.append(torch.matmul(torch.matmul(in_tensor[i][subgraphs_indexes[i]].to(device), self.lamda),in_tensor[i][subgraphs_indexes[i]].permute( 1, 0)).to(device))
            return torch.stack(adj_list)

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)


class GraphTransformerDecoder_FC(torch.nn.Module):
    def __init__(self,input,lambdaDim,SubGraphNodeNum, directed = True):
        super(GraphTransformerDecoder_FC, self).__init__()
        self.SubGraphNodeNum = SubGraphNodeNum
        self.directed = directed
        layers = [input] + [1024,1024, 1024]

        if directed:
            layers = layers + [SubGraphNodeNum * SubGraphNodeNum]
        else:
            layers = layers + [int((SubGraphNodeNum-1) * SubGraphNodeNum/2)+ SubGraphNodeNum]
        self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i+1],elementwise_affine=False) for i in range(len(layers) - 2)])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1],torch.float32) for i in range(len(layers) - 1)])

    def forward(self, in_tensor, subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.001)):

        for i in range(len(self.layers)):
            # in_tensor = self.normLayers[i](in_tensor)
            in_tensor = self.layers[i](in_tensor)
            if i !=len((self.layers))-1:
                in_tensor = activation(in_tensor)
                in_tensor = self.normLayers[i](in_tensor)
        if directed:
            ADJ = in_tensor.reshape(in_tensor.shape[0], self.SubGraphNodeNum, -1)
        else:
            ADJ = torch.zeros((in_tensor.shape[0],SubGraphNodeNum,SubGraphNodeNum)).to(in_tensor.device)
            ADJ[:,torch.tril_indices(SubGraphNodeNum,SubGraphNodeNum,-1)[0],torch.tril_indices(SubGraphNodeNum,SubGraphNodeNum,-1)[1]] = in_tensor[:, :(in_tensor.shape[-1])-SubGraphNodeNum]
            ADJ = ADJ + ADJ.permute(0,2,1)
            ind = np.diag_indices(ADJ.shape[-1])
            ADJ[:,ind[0], ind[1]] = in_tensor[:, -SubGraphNodeNum:] #torch.ones(ADJ.shape[-1]).to(ADJ.device)
        # adj_list= torch.matmul(torch.matmul(in_tensor, self.lamda),in_tensor.permute(0,2,1))
        # return adj_list
        # if subgraphs_indexes==None:
        # adj_list= torch.matmul(in_tensor,in_tensor.permute(0,2,1))
        return ADJ
        # else:
        #     adj_list = []
        #     for i in range(in_tensor.shape[0]):
        #         adj_list.append(torch.matmul(in_tensor[i][subgraphs_indexes[i]].to(device), in_tensor[i][subgraphs_indexes[i]].permute(0,2,1)).to(device))
        #     return torch.stack(adj_list)



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

        # if "RPF" in kernel_set:
        #     self.num_of_steps = ker.get("step_num")
        #     self.hist = Histogram(ker.get("bin_width"), ker.get("bin_center"))
        #
        # if "trans_matrix" in kernel_set:
        #     self.num_of_steps = ker.get("step_num")

        if "HistogramOfRandomWalks" in kernel_set:
            self.num_of_steps = ker.get("step_num")
            self.walks_hist = ker.get("HistogramOfRandomWalks")

        if "HistogramOfCycles" in kernel_set:
            self.Cycles_hist = Histogram(ker.get("RandomWalks_bin_width").to(device),
                                         ker.get("RandomWalks_bin_center").to(device))
            self.max_size_of_cyc = ker.get("max_size_of_cyc")
        if "ReachabilityInKsteps" in kernel_set:
            self.ReachabilityHist =Histogram(ker.get("RandomWalks_bin_width").to(device),
                                         ker.get("RandomWalks_bin_center").to(device))
            self.steps_of_reachability =ker.get("steps_of_reachability")

        if "clusteringCoefficient" in kernel_set:
            self.clusteringCoefficientHist = ker.get("clusteringCoefficientHist").to(device)

    def forward(self,adj):
        vec = self.kernel_function(adj)
        # return self.hist(vec)
        return vec

    def kernel_function(self, adj): # TODO: another var for keeping the number of moments
        # ToDo: here we assumed the matrix is symetrix(undirected) which might not
        vec = []  # feature vector
        for kernel in self.kernel_type:
            if "TotalNumberOfTriangles" == kernel:
                vec.append(self.TotalNumberOfTriangles(adj))
            if "in_degree_dist" == kernel:
                degree_hit = []
                adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
                for i in range(adj.shape[0]):
                    # degree = adj[i, subgraph_indexes[i]][:, subgraph_indexes[i]].sum(1).view(1, -1)

                    degree = adj_[i].sum(1).view(1, -1)
                    degree_hit.append(self.degree_hist(degree.to(device)))
                vec.append(torch.cat(degree_hit))
            if "out_degree_dist" == kernel:
                adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
                degree_hit = []

                for i in range(adj.shape[0]):
                    degree = adj_[i].sum(0).view(1, -1)
                    degree_hit.append(self.degree_hist(degree))
                vec.append(torch.cat(degree_hit))



            # if "RPF" == kernel:
            #     raise("should be changed") #ToDo: need to be fixed
            #     tr_p = self.S_step_trasition_probablity(adj, self.num_of_steps)
            #     for i in range(len(tr_p)):
            #         vec.append(self.hist(torch.diag(tr_p[i])))

            # if "trans_matrix" == kernel:
            #     vec.extend(self.S_step_trasition_probablity(adj, self.num_of_steps))
            #     # vec = torch.cat(vec,1)

            # if "tri" == kernel:  # compare the nodes degree in the given order
            #     tri, square = self.tri_square_count(adj)
            #     vec.append(tri), vec.append(square)
            #
            # if "TrianglesOfEachNode" == kernel: # this kernel returns a verctor, element i of this vector is the number of triangeles which are centered at node i
            #     vec.append(self.TrianglesOfEachNode(adj))
            #
            # if "ThreeStepPath" == kernel:
            #     vec.append(self.TreeStepPathes(adj))
            # [torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)]
            if "HistogramOfRandomWalks" == kernel:
                # remove diagonal elements
                eye = torch.eye(adj[0].shape[0]).to(device)
                adj_simple = adj *(1- eye)

                kth_walks = self.S_randomWalks(adj_simple,self.num_of_steps)
                for h_,walk in enumerate(kth_walks):
                    k_th_walk_hist = []
                    for i in range(adj_simple.shape[0]):
                        # k_th_walk_hist.append(self.walks_hist(torch.triu(walk[i], diagonal=1).view(1, -1).to(device)))
                        k_th_walk_hist.append(self.walks_hist[h_](walk[i][torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0],torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[1]].view(1, -1).to(device)))

                    vec.append(torch.cat(k_th_walk_hist))
                    # vec.append(walk[:, torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0],
                    # torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[1]].sum(-1))

            # if "cycles_number" == kernel:
            #     vec.extend(self.NumberOFCycleWithLenghtK(adj,3,self.num_of_steps+3))

            if "4-simple-cycle" == kernel:
                vec.append(self.NumberOf4Cycles(adj))

            if "HistogramOfCycles" == kernel:
                vec.extend(self.HistogramOfCycleWithLenghtK(adj,2,self.max_size_of_cyc))
            if "ReachabilityInKsteps" == kernel:
                vec.extend(self.ReachabilityInKsteps(adj))

            if "NumberOfVer" == kernel:
                vec.append(self.NumberOfVer(adj))

            if "twoStarMotifs" == kernel:
                vec.append(self.twoStarMotifs(adj))

            if "clusteringCoefficient" == kernel:
                vec.append(self.clusteringCoefficient(adj))

        return vec

    def tri_square_count(self, adj):
        eye = torch.eye(adj[0].shape[0]).to(device)
        simple_adj = adj * (1 - eye)
        two__ = torch.matmul(simple_adj, simple_adj)
        tri_ = torch.matmul(two__, simple_adj)
        squares = torch.matmul(two__, two__)
        return (torch.diagonal(tri_, dim1=1, dim2=2), torch.diagonal(squares, dim1=1, dim2=2))

    def twoStarMotifs(self, adj):
        adj = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
        return torch.matmul(adj, adj).sum((-2,-1)) -adj.sum((-2,-1))

    def clusteringCoefficient(self, adj):
            # adj = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
            # return (torch.matmul(adj,torch.matmul(adj, adj)).sum((-2, -1))/6)/(torch.matmul(adj, adj).sum((-2, -1)) - adj.sum((-2, -1))+1)

        adj = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
        # (torch.matmul(adj, torch.matmul(adj, adj)).sum((-2, -1)) / 6) / (
        #             torch.matmul(adj, adj).sum((-2, -1)) - adj.sum((-2, -1)) + 1)
        tri =    (torch.matmul(adj, torch.matmul(adj, adj)))
        tri = torch.diagonal(tri, dim1=1, dim2=2)
        degree = adj.sum(-1)
        all_possibles = degree * (degree - 1)
        all_possibles = torch.clamp(all_possibles, min=.001)
        clustering_coef = (tri / all_possibles) * (degree>.001)
        k_th_hist = []
        for i in range(clustering_coef.shape[0]):
                # k_th_walk_hist.append(self.walks_hist(torch.triu(walk[i], diagonal=1).view(1, -1).to(device)))
                k_th_hist.append(self.clusteringCoefficientHist(clustering_coef[i].view(1, -1)))

        k_th_hist = torch.cat(k_th_hist)
        return k_th_hist

    def S_randomWalks(self, adj, s=4, undirected = False ):
        """
         this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
        :param s: maximum step; the function will return s-step matrixes from 1 to s in a list
        :param Adj: adjacency matrix of the graph
        :return: a list in whcih the i-th elemnt is the i step transition probablity
        """
        # mask = torch.zeros(adj.shape).to(device)
        RW_list = []
        if undirected:
            p1 = torch.triu(adj, diagonal=1)
            RW_list.append(torch.matmul(p1, p1 ))
        else:
            # TP_list.append(torch.matmul(p1,p1))
            p1 = adj.to(device)
            RW_list.append( torch.matmul(p1, p1 ))

        for i in range(s-1):
            RW_list.append(torch.matmul(p1, RW_list[-1] ))
        return RW_list

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

    def TrianglesOfEachNode(self, adj,  ):
        """
         this method take an adjacency matrix and count the number of triangles centered at each node; this method return a vector for each graph
        """

        p1 = adj.to(device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)

        # to save memory Use ineficient loop
        tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
        return tri

    def NumberOf4Cycles(self, adj):
        adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
        A_two = torch.matmul(adj_, adj_)
        A_four = torch.matmul(A_two,A_two)
        A_two = A_two * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
        A_four = ((torch.diagonal(A_four,dim1=-2, dim2=-1).sum(-1)) - (adj_).sum([-2,-1]) - 2* A_two.sum([-2,-1]))/8
        return A_four

    def NumberOfVer(self, adj):
        adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
        return adj_.sum([-2,-1])


    def NumberOFCycleWithLenghtK(self,adj, k_l,k_u, Undirected = False):
        """
         this method take an adjacency matrix and count the number of simple cycle  with lenght [k_l,k_l+1,..., k_u]
        """
        result = []
        if not Undirected:
            p1 = adj *(1- torch.eye(adj.shape[-1], adj.shape[-1],device=device))
            Adj_k_th = p1
        else:
            p1 = torch.triu(adj, diagonal=1)
            Adj_k_th = p1
            p1 = p1.permute(0, 2, 1)


        for _ in range(1,k_l):
            Adj_k_th = torch.matmul(p1 ,Adj_k_th )

        for i in range(k_l,k_u):
            Adj_k_th = torch.matmul(p1, Adj_k_th )
            # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
            result.append((torch.diagonal(Adj_k_th,dim1=-2, dim2=-1)/(i*2)).sum(-1))

        return result #torch.cat(result)

    def HistogramOfCycleWithLenghtK(self,adj, k_l,k_u):
        """
         this method take an adjacency matrix and count the number of simple cycle  with lenght [k_l,k_l+1,..., k_u]
        """
        result = []
        p1 = adj *(1- torch.eye(adj.shape[-1], adj.shape[-1],device=device))
        Adj_k_th = p1
        for _ in range(1,k_l):
            Adj_k_th = torch.matmul(p1 ,Adj_k_th )

        for i in range(k_l,k_u):
            Adj_k_th = torch.matmul(p1, Adj_k_th )
            # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
            # result.append((torch.diagonal(Adj_k_th, dim1=-2, dim2=-1) / (i * 2)).sum(-1))
            result.append(self.Cycles_hist(torch.diagonal(Adj_k_th,dim1=-2, dim2=-1)/((i+1)*2)))
        return result

    def TreeStepPathes(self, adj,  ):
        """
         this method take an adjacency matrix and count the number of pathes between each two node with lenght 3; this method return a matrix for each graph
        """

        p1 = adj.to(device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)

        # to save memory Use ineficient loop
        # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
        tri = torch.matmul(p1, torch.matmul(p1, p1))
        return tri

    def TotalNumberOfTriangles(self, adj):
        """
         this method take an adjacency matrix and count the number of triangles in it the corresponding graph
        """
        p1 = adj.to(device)
        p1 = p1 * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)

        # to save memory Use ineficient loop
        tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
        return tri.sum(-1)

    def ReachabilityInKsteps(self, adj):
        """
         this method take an adjacency matrix and count the histogram of number of nodes which are ereachable from a node
        """
        result = []
        adj = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1], device=device))
        Adj_k_th = adj

        # result.append(self.ReachabilityHist(adj).sum(-1))
        for k_step in range(self.steps_of_reachability):
            Adj_k_th = torch.matmul(adj, Adj_k_th)
            reaches = Adj_k_th
            reaches = torch.clamp(reaches, min=0, max=1).sum(-1)

            result.append(self.ReachabilityHist(reaches))
        return result

def test_(number_of_samples, model ,graph_size, path_to_save_g, remove_self=True, save_graphs = True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    # model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    k = 0
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, model.embeding_dim]))
            z = torch.randn_like(z)
            start_time = time.time()

            adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            # sample_graph = sample_graph[:g_size,:g_size]
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
            if save_graphs:
                plotter.plotG(G, "generated" + dataset, file_name=f_name+"_ConnectedComponnents")
    # ======================================================
    # save nx files
    if save_graphs:
        nx_f_name = path_to_save_g + "_" + dataset + "_" + decoder_type + "_" + args.model + "_" + task
        with open(nx_f_name, 'wb') as f:
            pickle.dump(generated_graph_list, f)
    # # ======================================================
    return generated_graph_list

def EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated= True, _f_name=None):
    generated_graphs = test_(1, model , [x.shape[0] for x in test_list_adj], graph_save_path, save_graphs=Save_generated)
    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in generated_graphs]
    if Save_generated:
        np.save(graph_save_path+'generatedGraphs_adj_'+str(_f_name)+'.npy', graphs_to_writeOnDisk, allow_pickle=True)


    logging.info(mmd_eval(generated_graphs,[nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj]))
    print("====================================================")
    logging.info("====================================================")

    print("result for subgraph with maximum connected componnent")
    logging.info("result for subgraph with maximum connected componnent")
    generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if not nx.is_empty(G)]
    logging.info(mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj], diam= True))

    if Save_generated:
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in generated_graphs]
        np.save(graph_save_path+'Single_comp_generatedGraphs_adj_'+str(_f_name)+'.npy', graphs_to_writeOnDisk, allow_pickle=True)

        graphs_to_writeOnDisk = [G.toarray() for  G in test_list_adj]
        np.save(graph_save_path+'testGraphs_adj_.npy', graphs_to_writeOnDisk, allow_pickle=True)


def get_subGraph_features(org_adj, subgraphs_indexes, kernel_model):
    subgraphs =[]
    target_kelrnel_val = None

    for i  in range(len(org_adj)):
        subGraph = org_adj[i]
        if subgraphs_indexes!=None:
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i],:]
        # Converting sparse matrix to sparse tensor
        subGraph = torch.tensor(subGraph.todense())
        subgraphs.append(subGraph)
    subgraphs = torch.stack(subgraphs).to(device)

    if kernel_model!=None:
        target_kelrnel_val = kernel_model(subgraphs)
        target_kelrnel_val = [val.to("cpu") for val in target_kelrnel_val]
    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()
    return  target_kelrnel_val, subgraphs

# the code is a hard copy of https://github.com/orybkin/sigma-vae-pytorch
def log_guss(mean, log_std, samples):
    return 0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, num_nodes, alpha, reconstructed_adj_logit, pos_wight, norm, node_num ):

    # loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight)

    norm =    mean.shape[0] * mean.shape[1]
    kl = (1/norm)* -0.5 * torch.sum(1+2*log_std - mean.pow(2)-torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum()/float(reconstructed_adj.shape[0]*reconstructed_adj.shape[1]*reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []


    for i in range(len(target_kernel_val)):


        log_sigma  = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        each_kernel_loss.append(step_loss.cpu().detach().numpy()) #*alpha[i])
        kernel_diff += step_loss #*alpha[i]

    # kernel_diff += loss*alpha[-2]
    kernel_diff += kl * alpha[-1]
    # each_kernel_loss.append((loss*alpha[-2]).item())
    each_kernel_loss.append((kl * alpha[-1]).item())
    return kl , acc, kernel_diff, each_kernel_loss

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

list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True,limited_to=limited_to)#, _max_list_size=80)






if args.bfsOrdering==True:
    list_adj = BFS(list_adj)

# list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True, _max_list_size=80)

# list_adj, _ = permute(list_adj, None)
self_for_none = True
if (decoder_type)in  ("FCdecoder"):#,"FC_InnerDOTdecoder"
    self_for_none = True

if len(list_adj)==1:
    test_list_adj=list_adj.copy()
    list_graphs = Datasets(list_adj, self_for_none, list_x, None)
else:
    if task == "linkPrediction":
        max_size = max([x.shape[0] for x in list_adj])
        list_label = None
        raise ValueError("should be implemeted")
    else:
        max_size = None
    if task=="graphGeneration":
        list_label = None
        list_adj, test_list_adj, list_x_train , _ = data_split(list_adj,list_x)
        val_adj = list_adj[:int(len(test_list_adj))]
        list_graphs = Datasets(list_adj, self_for_none, list_x_train,list_label,Max_num=max_size, set_diag_of_isol_Zer=False)
    if task=="graphClasssification":
        list_graphs = Datasets(list_adj, self_for_none, list_x,list_label,Max_num=max_size)


print("#------------------------------------------------------")
fifty_fifty_dataset = list_adj + test_list_adj

fifty_fifty_dataset = [nx.from_numpy_matrix(graph.toarray()) for graph in fifty_fifty_dataset]
fifty_fifty_dataset = [ nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in fifty_fifty_dataset]

random.shuffle(fifty_fifty_dataset)
print("50%50 Evalaution of dataset")
if ideal_Evalaution:
    Diam_stats(fifty_fifty_dataset)
    logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset)/2)],fifty_fifty_dataset[int(len(fifty_fifty_dataset)/2):],diam=True))

del fifty_fifty_dataset
SubGraphNodeNum = subgraphSize if subgraphSize!=None else list_graphs.max_num_nodes
in_feature_dim = list_graphs.feature_size # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes


degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum, 1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

RandomWalks_bin_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
RandomWalks_bin_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum, 1)])
# degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
# degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum, 1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
#
# RandomWalks_bin_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
# RandomWalks_bin_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum, 1)])

HistogramOfRandomWalks = []
for i_ in range(step_num):
    i = i_+1#max(i_,1)
    # max_ =  max(int(list_graphs.get_max_degree()**i),SubGraphNodeNum) #SubGraphNodeNum#
    # max_ =  [SubGraphNodeNum, 900][i_]  # SubGraphNodeNum#
    # num_bin =
    max_ =  SubGraphNodeNum#
    binwith =int(max_/SubGraphNodeNum)# 1 #
    binSlope = .1
    bin_center = torch.tensor([[x] for x in range(0, max_, binwith)]).to(device)
    bin_width = torch.tensor([[binSlope] for x in range(0, max_, binwith)]).to(device)

    HistogramOfRandomWalks.append(Histogram(bin_width,bin_center, binwith))

bin_center = torch.tensor([[i/SubGraphNodeNum] for i in range(0, SubGraphNodeNum+1)]).to(device)
bin_width = torch.tensor([[.1] for i in bin_center]).to(device)
binwith =(1/SubGraphNodeNum)
clusteringCoefficientHist = Histogram(bin_width,bin_center, binwith)

kernel_model = kernel(kernel_type = kernl_type, step_num = step_num, steps_of_reachability = steps_of_reachability,
                      bin_width= bin_width,RandomWalks_bin_width=RandomWalks_bin_width, RandomWalks_bin_center=RandomWalks_bin_center, bin_center=bin_center, degree_bin_center=degree_center, degree_bin_width=degree_width, max_size_of_cyc = max_size_of_cyc, HistogramOfRandomWalks = HistogramOfRandomWalks, clusteringCoefficientHist=clusteringCoefficientHist)

if encoder_type =="AvePool":
    encoder = AveEncoder(in_feature_dim, [256], graphEmDim)
elif encoder_type =="GCNPool":
    encoder = GCNEncoder(in_feature_dim, [256, 256, 256], graphEmDim)
elif encoder_type == "diffPool":
    encoder = diffPoolEncoder(in_feature_dim, graphEmDim)

if decoder_type == "SBM":
    decoder = GraphTransformerDecoder(graphEmDim, 1024,nodeNum )
elif decoder_type == "FC":
    decoder = GraphTransformerDecoder_FC(graphEmDim, 256,nodeNum, directed )

model = kernelGVAE(kernel_model,encoder, decoder, AutoEncoder,graphEmDim=graphEmDim) # parameter namimng, it should be dimentionality of distriburion
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,6000,7000,8000,9000], gamma=0.5)

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
print(model)
logging.info(model.__str__())
min_loss = float('inf')

if(subgraphSize==None):
    list_graphs.processALL(self_for_none = self_for_none)
    adj_list = list_graphs.get_adj_list()
    graphFeatures , _ = get_subGraph_features(adj_list, None, kernel_model)
    list_graphs.set_features(graphFeatures)

# 50%50 Evaluation

test=False
if test==True:
    # fifty_fifty_dataset = list_adj + test_list_adj
    #
    # fifty_fifty_dataset = [nx.from_numpy_matrix(graph.toarray()) for graph in fifty_fifty_dataset]
    # random.shuffle(fifty_fifty_dataset)
    # print("50%50 Evalaution of dataset")
    # logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset)/2)],fifty_fifty_dataset[int(len(fifty_fifty_dataset)/2):]))
    # graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in fifty_fifty_dataset]
    # np.save(graph_save_path+dataset+'_dataset.npy', graphs_to_writeOnDisk, allow_pickle=True)

    #========================================
    model_dir = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_DD_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue100001651364417.4785793/"
    model.load_state_dict(torch.load(model_dir+"model_9999_3"))
    # EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )

# model_dir1 = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/"
# model.load_state_dict(torch.load(model_dir1+"model_9999_3"))
# EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )


for epoch in range(epoch_number):
    if epoch==30:
        print()
    list_graphs.shuffle()
    batch = 0
    for iter in range(0, max(int(len(list_graphs.list_adjs)/mini_batch_size),1)*mini_batch_size, mini_batch_size):
        from_ = iter
        to_= mini_batch_size*(batch+1)
    # for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
    #     from_ = iter
    #     to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)

        if subgraphSize==None:
            org_adj,x_s, node_num, subgraphs_indexes, target_kelrnel_val = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)
        else:
            org_adj,x_s, node_num, subgraphs_indexes = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)

        if(type(decoder))in [ GraphTransformerDecoder, GraphTransformerDecoder_FC]: #
            node_num = len(node_num)*[list_graphs.max_num_nodes]

        x_s = torch.cat(x_s)
        x_s = x_s.reshape(-1, x_s.shape[-1])

        model.train()
        if subgraphSize == None:
            _, subgraphs = get_subGraph_features(org_adj, None, None)
        else:
            target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)

        # target_kelrnel_val = kernel_model(org_adj, node_num)

        # batchSize = [org_adj.shape[0], org_adj.shape[1]]

        batchSize = [len(org_adj), org_adj[0].shape[0]]

        # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
        [graph.setdiag(1) for graph in org_adj]
        org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
        org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
        pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())

        reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val,reconstructed_adj_logit = model(org_adj_dgl.to(device), x_s.to(device), node_num, batchSize, subgraphs_indexes)
        kl_loss, acc, kernel_cost,each_kernel_loss = OptimizerVAE(reconstructed_adj, generated_kernel_val, subgraphs.to(device), [val.to(device) for val in target_kelrnel_val]  , post_log_std, post_mean, num_nodes, alpha,reconstructed_adj_logit, pos_wight, 2,node_num)


        loss = kernel_cost


        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), *each_kernel_loss],tmp, redraw= redraw)  # ["Accuracy", "loss", "AUC"])

        step+=1
        optimizer.zero_grad()
        loss.backward()

        if keepThebest and min_loss>loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()

        if (step+1) % visulizer_step == 0:

            model.eval()
            pltr.redraw()
            pltr.save_plot(graph_save_path + "KernelVGAE_log_plot")
            #----------------------------
            # Plot a reconstructed graph
            dir_generated_in_train = "generated_graph_train/"
            if not os.path.isdir(dir_generated_in_train):
                os.makedirs(dir_generated_in_train)
            rnd_indx = random.randint(0,len(node_num)-1)
            sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
            sample_graph = sample_graph[:node_num[rnd_indx],:node_num[rnd_indx]]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_matrix(sample_graph)
            plotter.plotG(G, "generated" + dataset, file_name=graph_save_path+"ReconstructedSample_At_epoch"+str(epoch))

            #---------------------------------------------------------------------------
            print("reconstructed graph vs Validation:")
            logging.info("reconstructed graph vs Validation:")
            reconstructed_adj = reconstructed_adj.cpu().detach().numpy()
            reconstructed_adj[reconstructed_adj >= 0.5] = 1
            reconstructed_adj[reconstructed_adj < 0.5] = 0
            reconstructed_adj = [nx.from_numpy_matrix(reconstructed_adj[i]) for i in range(reconstructed_adj.shape[0])]
            reconstructed_adj = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                 reconstructed_adj if not nx.is_empty(G)]

            target_set = [nx.from_numpy_matrix(val_adj[i].toarray()) for i in range(len(val_adj))]
            target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in target_set if
                          not nx.is_empty(G)]
            logging.info(mmd_eval(reconstructed_adj, target_set, diam=True))
            # ---------------------------------------------------------------------------

            if task=="graphGeneration":
                EvalTwoSet(model, val_adj, graph_save_path, Save_generated= True,_f_name=epoch)

                if ((step+1) % visulizer_step*2):
                    torch.save(model.state_dict(), graph_save_path+"model_"+str(epoch)+"_"+str(batch))
            model.train()

        k_loss_str=""
        for indx,l in enumerate(each_kernel_loss):
            k_loss_str+=functions[indx+2]+":"
            k_loss_str+=str(l)+".   "

        print("Epoch: {:03d} |Batch: {:03d} | loss: {:05f} |  z_kl_loss: {:05f} | accu: {:03f}".format(
            epoch + 1,batch,  loss.item(), kl_loss.item(), acc),k_loss_str)
        logging.info("Epoch: {:03d} |Batch: {:03d} | loss: {:05f}  | z_kl_loss: {:05f} | accu: {:03f}".format(
            epoch + 1,batch,  loss.item(),  kl_loss.item(), acc) +" "+ str(k_loss_str))
        batch+=1
        # scheduler.step()

model.eval()
torch.save(model.state_dict(), graph_save_path+"model_"+str(epoch)+"_"+str(batch))

stop = timeit.default_timer()
print("trainning time:", str(stop-start))
logging.info("trainning time: "+ str(stop-start))
# save the train loss for comparing the convergence
import json

file_name = graph_save_path+"_"+encoder_type+"_"+decoder_type+"_"+dataset+"_"+task+"_"+args.model+"_elbo_loss.txt"

with open(file_name, "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2])+np.array(pltr.values_train[-1])), fp)


with open(file_name+"_CrossEntropyLoss.txt", "w") as fp:
    json.dump(list(np.array(pltr.values_train[-2])), fp)


with open(file_name+"_train_loss.txt", "w") as fp:
    json.dump(pltr.values_train[1], fp)


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







# graph Classification
if task == "graphClasssification":


    org_adj,x_s, node_num, subgraphs_indexes,  labels = list_graphs.adj_s, list_graphs.x_s, list_graphs.num_nodes, list_graphs.subgraph_indexes, list_graphs.labels

    if(type(decoder))in [ GraphTransformerDecoder, GraphTransformerDecoder_FC]: #
        node_num = len(node_num)*[list_graphs.max_num_nodes]

    x_s = torch.cat(x_s)
    x_s = x_s.reshape(-1, x_s.shape[-1])

    model.eval()
    # if subgraphSize == None:
    #     _, subgraphs = get_subGraph_features(org_adj, None, None)

    batchSize = [len(org_adj), org_adj[0].shape[0]]

    [graph.setdiag(1) for graph in org_adj]
    org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]

    org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
    mean, std = model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
    # prior_samples = model.reparameterize(mean, std, node_num)
    # model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
    # _, prior_samples, _, _, _,_ = model(org_adj_dgl.to(device), x_s.to(device), node_num, batchSize, subgraphs_indexes)



    import classification as CL

    # NN Classifier
    labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.NN(mean.cpu().detach(), labels)

    print("Accuracy:{}".format(accuracy),
          "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
          "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
          "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
          "confusion matrix:{}".format(conf_matrix))

    # KNN clasiifier
    labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.knn(prior_samples.cpu().detach(), labels)
    print("Accuracy:{}".format(accuracy),
          "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
          "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
          "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
          "confusion matrix:{}".format(conf_matrix))
# evaluatin graph statistics in graph generation tasks


if task=="graphGeneration":
    EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated= True,_f_name="final_eval")