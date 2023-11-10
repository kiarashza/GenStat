
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
import os
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


parser.add_argument('-e', dest="epoch_number" , default=10000, help="Number of Epochs", type=int)
parser.add_argument('-v', dest="Vis_step", default=1000, help="model learning rate")
# parser.add_argument('-AE', dest="AutoEncoder", default=True, help="either update the log plot each step")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.001, help="model learning rate",type=float) # henerally   0.0003 works well # 0005 for pca+ diagonal varianve
parser.add_argument('-dataset', dest="dataset", default="lobster", help="possible choices are:  mnist_01_percent,random_powerlaw_tree, PVGAErandomGraphs_dev, PVGAErandomGraphs, IMDBBINARY, IMDbMulti,wheel_graph, star, triangular_grid, multi_community, NCI1, ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")#citeceer: ego; DD:protein
parser.add_argument('-NofCom', dest="num_of_comunities", default=256, help="Number of comunites")
parser.add_argument('-graphEmDim', dest="graphEmDim", default=1024, help="the simention of graphEmbedingLAyer")
parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
parser.add_argument('-graph_save_path', dest="graph_save_path", default=None, help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature" , default=True, help="either use features or identity matrix")
parser.add_argument('-Split', dest="split_the_data_to_train_test" , default=True, help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-PATH', dest="PATH" , default="model", help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder" , default="FC", help="the decoder type, FC or SBM")
parser.add_argument('-encoder', dest="encoder_type" , default="AvePool", help="the encoder")    #"diffPool" "AvePool" "GCNPool"
parser.add_argument('-batchSize', dest="batchSize" , default=100, help="the size of each batch")
parser.add_argument('-UseGPU', dest="UseGPU" , default=True, help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model" , default="PMI", help=" PMI")
parser.add_argument('-device', dest="device" , default="cuda:0", help="Which device should be used")
parser.add_argument('-task', dest="task" , default="graphGeneration", help="linkPrediction, GraphCompletion, graphClasssification or graphGeneration")
parser.add_argument('-BFS', dest="bfsOrdering" , default=True, help="use bfs for graph permutations", type=bool)
parser.add_argument('-directed', dest="directed" , default=False, help="is the dataset directed?!", type=bool)
parser.add_argument('-beta', dest="beta" , default=None, help="beta coefiicieny", type=float)
parser.add_argument('-limited_to', dest="limited_to" , default=None, help="How many instance you want to pick, its for reducing trainning time in dev", type=int)
parser.add_argument('-ideal_Evalaution', dest="ideal_Evalaution" , default=False, help="if you want to comapre the 50%50 subset of dataset comparision?!", type=bool)
parser.add_argument("-desc_fun", dest="desc_fun" , default= ["in_degree_dist","HistogramOfRandomWalks", "ReachabilityInKsteps","HistogramOfCycles"], help="the descriptor function which should be used", nargs='+',)
# kernl_type = ["TotalNumberOfTriangles", "HistogramOfRandomWalks","in_degree_dist", "out_degree_dist", "ReachabilityInKsteps","HistogramOfCycles"] # "clusteringCoefficient", "cycles_number","ReachabilityInKsteps" ReachabilityInKsteps "steps_of_reachability"
parser.add_argument('-write_them_in', dest="write_them_in" , default=None, help="path_to write generated/test graphs", )
parser.add_argument('-PCATransformer', dest="PCATransformer" , default=False, help="either use PCA transformer or net", )
parser.add_argument('-rand_strategy', dest="rand_strategy" , default=False, help="either use rand matrix instead of  PCA componnents", )
parser.add_argument('-ResidualAnlyze', dest="ResidualAnlyze" , default=False, help="either save the residuls for feuture znzlyze or not", )
parser.add_argument('-EvalOnTest', dest="EvalOnTest" , default=False, help="either print the statistics based EVAL on the test set", )
parser.add_argument('-binningStrategy', dest="binningStrategy" , default=["EqualWidth","BounderBinner"], help="binning schema: UniformBinner, EqualWidth or EqualFreq", nargs='+' )
parser.add_argument('-single_variance', dest="single_variance" , default=True, help="use a shared variancee for each invarients, othervise use a variance for each dim", )
parser.add_argument('-membershipFunction', dest="membershipFunction" , default="gaussian", help="soft membership function; either tri(triangle) or gaussian", )
parser.add_argument('-scale', dest="scale" , default=False, help="either scale an invaient dimenssion or not as preprosseing of PCA", )
parser.add_argument('-bin_num', dest="bin_num" , default="sqrt(N)", help="number of bins: eitehr sqrt(N) or N", )


args = parser.parse_args()
# torch.autograd.set_detect_anomaly(True)


single_variance = args.single_variance
MembershipFunction = args.membershipFunction
binningStrategy = args.binningStrategy
EvalOnTest = args.EvalOnTest
encoder_type = args.encoder_type
ResidualAnlyze = args.ResidualAnlyze
ideal_Evalaution = args.ideal_Evalaution
graphEmDim = args.graphEmDim
visulizer_step = args.Vis_step
PCATransformer = args.PCATransformer
redraw = args.redraw
device = args.device
task = args.task
rand_strategy = args.rand_strategy
limited_to = args.limited_to
directed = args.directed
epoch_number = args.epoch_number
bin_num = args.bin_num
lr = args.lr
scale = args.scale
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
write_them_in = args.write_them_in
split_the_data_to_train_test = args.split_the_data_to_train_test
kernl_type = args.desc_fun

if visulizer_step == None:
    visulizer_step = min(10000,epoch_number-2)
if graph_save_path==None:
    graph_save_path = "MMD_"+encoder_type+"_"+decoder_type+"_"+dataset+"_"+task+"_"+args.model+"BFS"+str(args.bfsOrdering)+str(args.epoch_number) +str(time.time())+"/"
from pathlib import Path
Path(graph_save_path).mkdir(parents=True, exist_ok=True)
if limited_to==-1:
    limited_to=None

# maybe to the beest way
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=graph_save_path+'log.log', filemode='w',  level=logging.INFO)





# **********************************************************************
# setting
print("KernelVGAE SETING: "+str(args))
logging.info("KernelVGAE SETING: "+str(args))
PATH = args.PATH # the dir to save the with the best performance on validation data


if  dataset == "large_grid":
    alpha = [ 1]
elif dataset == "small_triangular_grid":
    alpha = [ 1]
elif dataset == "ogbg-molbbbp":
    alpha = [ 1]
elif dataset == "PTC":
    alpha = [ 1]
elif dataset == "PVGAErandomGraphs_dev"or dataset == "PVGAErandomGraphs" or dataset == "PVGAErandomGraphs_10000" or dataset=="100_PVGAErandomGraphs":
    alpha = [1]
    max_size_of_cyc = 3
    steps_of_reachability = 4
    step_num = 4
elif dataset == "FIRSTMM_DB":
    alpha = [1]
elif dataset == "DD":
    alpha = [1]
    max_size_of_cyc = 3
    steps_of_reachability = 4
    step_num = 4
elif dataset=="grid" or dataset == "small_grid":
    alpha = [ 1]
elif dataset == "IMDbMulti" or dataset == "IMDBBINARY":
    alpha = [1]
    max_size_of_cyc = 3
    steps_of_reachability = 2
    step_num = 2

elif dataset=="lobster":
    max_size_of_cyc = 3
    steps_of_reachability = 4
    step_num = 2
    alpha = [1]
elif dataset == "wheel_graph":
    alpha = [ 200000000]
elif dataset == "triangular_grid":
    alpha = [1]
elif dataset == "random_powerlaw_tree":
    alpha = [1]
    max_size_of_cyc = 3
    steps_of_reachability = 4
    step_num = 4
else:
    warnings.warn("Step num is not tuned; using the default values")
    alpha = [1]
    max_size_of_cyc = 3
    steps_of_reachability = 2
    step_num = 2


#------------------------------------------------------------------------
# if args.model == "kipf":
#     alpha= [1]
#     step_num = 0
AutoEncoder = False
if task=="graphClasssification":
    AutoEncoder = True

if AutoEncoder == True:
    alpha[-1] = 0

if task=="GraphCompletion":
    alpha[-1]=0

if args.beta!=None:
    alpha[-1] = args.beta
#------------------------------------------------------------------------
max_needed_step = max([max_size_of_cyc,steps_of_reachability,step_num])
print("alpha: "+str(alpha) +"   max_size_of_cyc,steps_of_reachability, step_num:"+str([max_size_of_cyc,steps_of_reachability,step_num]), "  binningStrategy", binningStrategy)


logging.info("kernl_type:"+str(kernl_type))
logging.info("alpha: "+str(alpha))


bin_center = torch.tensor([[x / 10000] for x in range(0, 1000, 1)])
bin_width = torch.tensor([[9000] for x in range(0, 1000, 1)])# with is propertion to revese of this value;

device = torch.device(device if torch.cuda.is_available() and use_gpu else "cpu")
print("the selected device is :", device)
logging.info("the selected device is :"+ str(device))

#========================================================================
# setting the plots legend
functions= ["Accuracy", "loss"]
# if args.model == "kernel" or args.model =="KernelAugmentedWithTotalNumberOfTriangles":
    # functions.extend(["Kernel"+str(i) for i in range(step_num)])
for ker in kernl_type:
    if ker == "HistogramOfCycles":
        functions.extend(["HistogramOfCycles_" + str(i+3) for i in range(max_size_of_cyc-2)])
        # functions.extend(["NumbetOfCycles_" + str(i + 3) for i in range(max_size_of_cyc - 2)])
    elif ker == "HistogramOfRandomWalks":
        functions.extend(["RandWalksHist_" + str(i+2)+"_steps" for i in range(step_num)])
        # functions.extend(["NumberOfRandWalks_" + str(i) for i in range(step_num)])
    elif ker == "cycles_number":
        functions.extend(["CyclHist" + str(i) for i in range(step_num)])
    elif ker == "ReachabilityInKsteps":
        functions.extend(["ReachabilityIn_" + str(i+2) +"_steps" for i in range(steps_of_reachability)])
    else:
        functions.append(ker)
functions.append("KL-D")

#========================================================================



pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log",functions=functions)


split_the_data_to_train_test = False



def PCA_Analiz(data, plot_scree=True, n_components=None):
    from sklearn.decomposition import PCA
    from scipy import stats
    # cor = np.cov(np.transpose(data))
    # person_corr = numpy.corrcoef(np.transpose(data))
    pca = PCA(n_components)
    pca.fit(data)
    print("Cumulative Variances (Percentage):")
    print(np.cumsum(pca.explained_variance_ratio_ * 100))

    if plot_scree:
        components = len(pca.explained_variance_ratio_)
        np.cumsum(pca.explained_variance_ratio_ * 100)
        import matplotlib.pyplot as plt
        plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))
        plt.xlabel("Number of components")
        plt.ylabel("Explained variance (%)")

def PCA_projector(data, n_components=.99, whiten=False, normalize=True):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    # cor = np.cov(np.transpose(data))
    # person_corr = numpy.corrcoef(np.transpose(data))
    scaler = StandardScaler(with_mean=False)
    scaler.fit(data)
    pca = PCA(n_components,whiten=whiten)
    if normalize:
        pca.fit(data/scaler.var_)
    else:
        pca.fit(data )
    print("Cumulative Variances (Percentage):")
    print(np.cumsum(pca.explained_variance_ratio_ * 100))
    print("each feature var", scaler.var_)

    return pca, scaler

def PCA_transform(X, TransformerMatrix, Mean, Variance):
    """Apply dimensionality reduction to X."""
    device = X.device
    if Mean is not None:
        X = X - Mean.to(device)
    if  hasattr(Variance,"shape"):
        X /= Variance.to(device)
    X_transformed = torch.matmul(X,TransformerMatrix.to(device))

    return X_transformed

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
        # self.normLayers = torch.nn.ModuleList(
        #     [torch.nn.BatchNorm1d(hiddenLayers[i + 1]) for i in range(len(hiddenLayers) - 1)])
        # self.normLayers.append(torch.nn.BatchNorm1d(GraphLatntDim))
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

class Binner():
    # def EqualWidthBinner( featureValues, numBins, MemFunc, minZero=True, UpperBound=None, ):
    #     featureValues = featureValues.reshape(-1)
    #     min_ = 0
    #     if not minZero:
    #         min_ = torch.min(featureValues)
    #
    #     max_ = torch.max(featureValues).detach().cpu().item()
    #
    #     bin_width = []
    #     bin_center = []
    #     slope = []
    #     step = (((max_ - min_)/numBins))
    #     if step !=0:
    #         for i in range(numBins+1):
    #             bin_center+=[min_+step*i]
    #             bin_width += [step]
    #         if MemFunc=="tri":
    #             slope_ = max(.1, 1 / (len(bin_width)))  # a hyperparamer to chose
    #         else:
    #             slope_ = max(.33, 1 / (len(bin_width)))  # a hyperparamer to chose
    #         slope = [slope_ for x in range(0, len(bin_center), 1)]
    #     else :
    #         if UpperBound==None:
    #             print("You need upper bound in this case")
    #         bin_width += [UpperBound]
    #         bin_center += [min_]
    #         if MemFunc=="tri":
    #             slope+=[1.]
    #         else:
    #             slope += [2.]
    #
    #     if UpperBound!=None:
    #         step = (UpperBound - bin_center[-1])
    #         bin_center += [UpperBound]
    #         bin_width += [UpperBound]
    #         if MemFunc=="tri":
    #             slope.append(1.)
    #         else:
    #             slope.append(2.)
    #
    #
    #     bin_width =  torch.tensor(bin_width)/torch.tensor(slope)
    #     return torch.tensor(bin_center), bin_width#, torch.tensor(slope)


    # def EqualFreqBinner(featureValues, numBins, MemFunc, UpperBound=None):
    #
    #     featureValues = featureValues.reshape(-1)
    #
    #     bin_width = []
    #     bin_center = []
    #     # pre_binCenter = 0
    #     featureValues = np.sort(featureValues.detach().cpu().numpy())
    #     for b in range(numBins):
    #
    #         numElementsInEachBin = int(len(featureValues)/(numBins-b))-1
    #         binCenter = np.mean(featureValues[featureValues<=featureValues[numElementsInEachBin]])
    #         # binWidth = binCenter - pre_binCenter
    #         # pre_binCenter = binCenter
    #         bin_center += [binCenter]
    #         # bin_width += [binWidth]
    #         featureValues = featureValues[featureValues>featureValues[numElementsInEachBin]]
    #         if len(featureValues)==0:
    #             break
    #     for i,_ in enumerate(bin_center):
    #         left =0 if i==0 else bin_center[i-1]
    #         right =0 if i==(len(bin_center)-1) else bin_center[i+1]
    #         if left==0 and right==0:
    #             if MemFunc == "tri":
    #                 bin_width=[UpperBound]
    #             else:
    #                 bin_width=[UpperBound]
    #
    #         else: bin_width+=[max(bin_center[i]-left, right-bin_center[i])]
    #     # attach the bin for the uppen bound
    #     if UpperBound!=None:
    #         # step = (UpperBound - bin_center[-2])
    #         bin_center += [UpperBound]
    #         bin_width += [UpperBound]
    #     if len(bin_center)>2:
    #         bin_center = [0.] + bin_center
    #         bin_width = [bin_width[-1]] + bin_width
    #
    #     if MemFunc == "tri":
    #         slope = torch.tensor([.1 for x in range(0, len(bin_center), 1)])
    #         slope[-1] = 1
    #         bin_width = torch.tensor(bin_width)/slope
    #
    #     else:
    #         slope = torch.tensor([2. for x in range(0, len(bin_center), 1)])
    #         slope[-1] = 2.
    #         bin_width = torch.tensor(bin_width)/slope
    #
    #     return torch.tensor(bin_center), (bin_width)#, torch.tensor(slope)
    def EqualWidthBinner(featureValues, numBins, minZero=True, UpperBound = None ):
        featureValues = featureValues.reshape(-1)
        min_ = 0
        if  not minZero:
            min_ = torch.min(featureValues)
        # if UpperBound==None:
        #     max_ = torch.max(featureValues).detach().cpu().item()
        # else:
        #     max_= UpperBound
        max_ = torch.max(featureValues).detach().cpu().item()

        bin_width = []
        bin_center = []

        step = (((max_ - min_) / numBins))
        if step != 0:
            for i in range(numBins + 1):
                bin_center += [min_ + step * i]
                bin_width += [step]
        else:
            bin_center += [min_]
            bin_width += [torch.tensor(1)]

        # if UpperBound!=None:
        #     # step = (UpperBound - bin_center[-2])
        #     bin_center += [UpperBound]
        #     bin_width += [UpperBound]
        #
        #     bin_center = [0.] + bin_center
        #     bin_width = [bin_width[-1]] + bin_width

        slope = [1. for i in bin_width] # hyper-parametr indicate the bins overlap

        bin_width = torch.tensor(bin_width) / torch.tensor(slope)
        return torch.tensor(bin_center), bin_width  # , torch.tensor(slope)


    def EqualFreqBinner(featureValues, numBins, UpperBound=None):

        featureValues = featureValues.reshape(-1)

        bin_width = []
        bin_center = []
        # pre_binCenter = 0
        featureValues = np.sort(featureValues.detach().cpu().numpy())
        numElementsInEachBin = int(len(featureValues) / (numBins))
        for b in range(numBins):

            numElementsInEachBin = int(len(featureValues) / (numBins))
            # bin = (featureValues[featureValues<=featureValues[numElementsInEachBin]])
            bin = featureValues if b==(numBins-1) else featureValues[featureValues<=featureValues[numElementsInEachBin]]
            # binCenter = np.mean([np.max(bin), np.min(bin)])
            # binWidth = np.max(bin) - np.mean([np.max(bin), np.min(bin)])
            #
            binCenter = np.mean(bin)
            binWidth = np.std(bin)

            # pre_binCenter = binCenter
            if (len(bin_center)==0 or binCenter!=bin_center[-1]):
                bin_center += [binCenter]
                bin_width += [binWidth]
            # featureValues = featureValues[numElementsInEachBin:]
            featureValues = featureValues[featureValues>featureValues[numElementsInEachBin]]
            if len(featureValues)==0:
                break

        deflt = 1
        for i,_ in enumerate(bin_width):
            if bin_width[i]==0:
                bin_width[i] = deflt
        # attach the bin for the uppen bound
        # if UpperBound!=None:
        #     # step = (UpperBound - bin_center[-2])
        #     bin_center += [UpperBound]
        #     bin_width += [UpperBound]
        #
        #     bin_center = [0.] + bin_center
        #     bin_width = [bin_width[-1]] + bin_width

        slope = torch.tensor([1. for x in range(0, len(bin_center), 1)])
        slope[-1] = 2.
        bin_width = torch.tensor(bin_width)/slope

        return torch.tensor(bin_center), (bin_width)#, torch.tensor(slope)

    def BounderBinner( lowerBound, UpperBound):
        bin_width = []
        bin_center = []
        # pre_binCenter = 0

        if UpperBound!=None:
            # step = (UpperBound - bin_center[-2])
            bin_center += [UpperBound]
            bin_width += [UpperBound]

            bin_center = [0.] + bin_center
            bin_width = [bin_width[-1]] + bin_width

        slope = torch.tensor([2. for x in range(0, len(bin_center), 1)])
        bin_width = torch.tensor(bin_width)/slope

        return torch.tensor(bin_center), (bin_width)


    def combineBinner(self, binners, featureValues, numBins,  UpperBound=None,):
        center  = []
        width = []
        for binner in binners:
            if binner== Binner.BounderBinner:
                _center, _width = binner(0 , UpperBound)
            else:
                _center, _width = binner(featureValues, numBins, UpperBound=UpperBound)
            center.append(_center)
            width.append(_width)

        return torch.cat(center), torch.cat(width)

        # This schema use both equl freq and equalwith

    def UniformBinner(feature_values, numBins, UpperBound):
        min_ = 0

        max_ = UpperBound

        bin_width = []
        bin_center = []

        step = (max_ - min_)/numBins
        for i in range(numBins+1):
            bin_center+=[min_+step*i]
            bin_width += [step]

        # slope = torch.tensor([[.1] for x in range(0, len(bin_center), 1)])
        # # slope[-1] = .5
        return torch.tensor(bin_center), torch.tensor(bin_width)#, torch.tensor(slope)

class Histogram(torch.nn.Module):
    # this is a soft histograam Function.
    #for deails check section "3.2. The Learnable Histogram Layer" of
    # "Learnable Histogram: Statistical Context Features for Deep Neural Networks"
    def __init__(self, bin_centers = None, bin_width = None, MembershipFunction="tri"):
        super(Histogram, self).__init__()
        self.bin_width = bin_width.to(device).reshape(1,-1,1)
        self.bin_center = bin_centers.to(device).reshape(1,-1,1)
        self.MemberFun = MembershipFunction
        # self.norm = torch.tensor(normalizer).to(device).reshape(1,-1,1)
        if self.bin_width == None:
            self.prism()
        else:
            self.bin_num = self.bin_width.shape[0]

    def forward(self, vec): # id guassian==False it will be triangle membership fun
        if self.MemberFun=="gaussian":
            score_vec = (vec.view(vec.shape[0], 1, vec.shape[-1]) - self.bin_center) / self.bin_width
            score_vec = torch.square(score_vec)*-.5
            score_vec = torch.exp(score_vec)
            return score_vec.sum(-1)
        #REceive a vector and return the soft histogram

        elif self.MemberFun=="tri":
            #comparing each element with each of the bin center
            score_vec = (vec.view(vec.shape[0],1,vec.shape[-1]) -  self.bin_center)/self.bin_width
            # score_vec = vec-self.bin_center
            score_vec = 1-torch.abs(score_vec)
            score_vec = torch.relu(score_vec)
            # mask = (vec >= self.bin_center[-1][0])
            # vec = vec * mask
            # vec = vec/self.bin_center[-1][0]
            # vec = vec.sum(-1, keepdim=True)
            ## vec= vec.clamp(0,1).sum(-1, keepdim=True)
            # return torch.cat((score_vec.sum(2),vec),1)

            return score_vec.sum(-1)
        else:
            raise Exception("Sorry, Membership function is not defined")

class GraphTransformerDecoder(torch.nn.Module):
    def __init__(self,input,lambdaDim,SubGraphNodeNum, layers= [128 ]):
        super(GraphTransformerDecoder, self).__init__()
        self.lamda =torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
        self.SubGraphNodeNum = SubGraphNodeNum
        layers = [input] + [2048, 1024]+[lambdaDim*SubGraphNodeNum]
        self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i+1],elementwise_affine=False) for i in range(len(layers) - 2)])
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

        adj_list= torch.matmul(torch.matmul(in_tensor, self.lamda),in_tensor.permute(0,2,1))
        return adj_list
        # if subgraphs_indexes==None:
        #     adj_list= torch.matmul(torch.matmul(in_tensor, self.lamda),in_tensor.permute(0,2,1))
        #     return adj_list
        # else:
        #     adj_list = []
        #     for i in range(in_tensor.shape[0]):
        #         adj_list.append(torch.matmul(torch.matmul(in_tensor[i][subgraphs_indexes[i]].to(device), self.lamda),in_tensor[i][subgraphs_indexes[i]].permute( 1, 0)).to(device))
        #     return torch.stack(adj_list)

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
        # self.normLayers = torch.nn.ModuleList(
        #     [torch.nn.BatchNorm1d(layers[i + 1], ) for i in range(len(layers) - 2)])
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
            self.degree_hist = ker.get("degree_hist")

        if "HistogramOfRandomWalks" in kernel_set:
            self.num_of_steps = ker.get("step_num")
            self.walks_hist = ker.get("HistogramOfRandomWalks")

        if "HistogramOfCycles" in kernel_set:
            self.Cycles_hist = ker.get("cyclesHistogram")
            self.max_size_of_cyc = ker.get("max_size_of_cyc")
        if "ReachabilityInKsteps" in kernel_set:
            self.ReachabilityHist =ker.get("ReachabilityInKsteps")
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
        eye = torch.eye(adj[0].shape[0]).to(device)
        adj = adj * (1 - eye)

        for kernel in self.kernel_type:
            if "TotalNumberOfTriangles" == kernel:
                vec.append(self.TotalNumberOfTriangles(adj))
            if "in_degree_dist" == kernel:
                # degree_hit = []
                adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
                # # adj_ = adj_.permute(0,2,1)
                # for i in range(adj.shape[0]):
                #     # degree = adj[i, subgraph_indexes[i]][:, subgraph_indexes[i]].sum(1).view(1, -1)
                #
                #     degree = adj_[i].sum(-1).view(1, -1)
                #     degree_hit.append(self.degree_hist(degree))
                # vec.append(torch.cat(degree_hit))
                vec.append(self.degree_hist(adj_.sum(-1)))
            if "out_degree_dist" == kernel:
                adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
                # degree_hit = []
                #
                # for i in range(adj.shape[0]):
                #     degree = adj_[i].sum(0).view(1, -1)
                #     degree_hit.append(self.degree_hist(degree))
                vec.append(self.degree_hist(adj_.sum(0)))



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

        adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1])).to(device)
        # (torch.matmul(adj, torch.matmul(adj, adj)).sum((-2, -1)) / 6) / (
        #             torch.matmul(adj, adj).sum((-2, -1)) - adj.sum((-2, -1)) + 1)
        tri =    (torch.matmul(adj_, torch.matmul(adj_, adj_)))
        tri = torch.diagonal(tri, dim1=1, dim2=2)
        degree = adj_.sum(-1)
        all_possibles = degree * (degree - 1)

        # all_possibles = torch.clamp(all_possibles, min=.001)
        mask = all_possibles<=0
        mask = torch.tensor(mask)+0
        all_possibles = all_possibles + mask
        clustering_coef = (tri / all_possibles) #* (degree>.001)
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
        counter=0
        for i in range(k_l,k_u):
            Adj_k_th = torch.matmul(p1, Adj_k_th )
            # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
            # result.append((torch.diagonal(Adj_k_th, dim1=-2, dim2=-1) / (i * 2)).sum(-1))
            result.append(self.Cycles_hist[counter](torch.diagonal(Adj_k_th,dim1=-2, dim2=-1)/((i+1)*2)))#((i+1)*2)
            counter+=1
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
        adj_ = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1], device=device))
        Adj_k_th = adj_

        # result.append(self.ReachabilityHist(adj).sum(-1))
        for k_step in range(self.steps_of_reachability):
            # Adj_k_th = torch.matmul(adj, Adj_k_th)
            # reaches = Adj_k_th
            # reaches = torch.clamp(reaches, min=0, max=1).sum(-1)#result.append(self.ReachabilityHist(Adj_k_th.sum(-1)))
            Adj_k_th = torch.matmul(adj_, Adj_k_th)
            Adj_k_th = torch.clamp(Adj_k_th, max=1)
            result.append(self.ReachabilityHist[k_step](Adj_k_th.sum(-1)))
        return result

def graphPloter(listOfNXgraphs, dir):
    for i, G in enumerate(listOfNXgraphs):
            plotter.plotG(G, file_name=dir+"_"+str(i))


def graph_generator( model, batch_size, num_graphs=None, remove_self=True):

    model.eval()
    generated_graph_list = []
    if num_graphs==None:
        num_graphs = batch_size

    while len(generated_graph_list) <num_graphs:
            z = torch.tensor(numpy.random.normal(size=[batch_size, model.embeding_dim]))
            z = torch.randn_like(z)
            start_time = time.time()

            adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graphs = reconstructed_adj.cpu().detach().numpy()
            # sample_graph = sample_graph[:g_size,:g_size]
            sample_graphs[sample_graphs >= 0.5] = 1
            sample_graphs[sample_graphs < 0.5] = 0

            for graph in sample_graphs:
                G = nx.from_numpy_matrix(graph)
                if remove_self:
                    G.remove_edges_from(nx.selfloop_edges(G))
                G.remove_nodes_from(list(nx.isolates(G)))
                generated_graph_list.append(G)
    return generated_graph_list[:num_graphs]

def EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated= True, _f_name=None, OnlyMax_Con_com=True):
    generated_graphs = graph_generator(model , mini_batch_size, len(test_list_adj))
    logs = ""
    if OnlyMax_Con_com==False:
        print("all connected_componnents:")
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs[:100]]
        if Save_generated:
            np.save(graph_save_path + 'generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                    allow_pickle=True)
        logs = " connected_componnents: "
        log = mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
        logs += log
        logging.info(log)
        print("====================================================")
    logging.info("====================================================")
    logs+= "\n"
    print("result for subgraph with maximum connected componnent")
    logging.info("result for subgraph with THE maximum connected componnent")
    logs += "THE maximum_connected_componnent: "
    generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if not nx.is_empty(G)]
    log = mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj], diam= True)
    logging.info(log)
    logs+=log
    if Save_generated:
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in generated_graphs[:100]]
        np.save(graph_save_path+'Max_Con_comp_generatedGraphs_adj_'+str(_f_name)+'.npy', graphs_to_writeOnDisk, allow_pickle=True)
        graphPloter(generated_graphs, dir =graph_save_path, )

        graphs_to_writeOnDisk = [G.toarray() for  G in test_list_adj[:100]]
        np.save(graph_save_path+'_Target_Graphs_adj_.npy', graphs_to_writeOnDisk, allow_pickle=True)
    return logs

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
    if len(samples.shape)==1:
        return (0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi))
    else:
        if single_variance:
             return (0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)).sum([i+1 for i in range(len(samples.shape)-1)])
        else: return (0.5 * torch.pow((samples - mean) / log_std.exp(), 2)  + 0.5 * np.log(2 * np.pi)).sum([i+1 for i in range(len(samples.shape)-1)]) + log_std.sum()
        # return (0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)).sum([i + 1 for i in range(len(samples.shape) - 1)])


def mmdBylabel(reconstructed_graphs, train_set, labels, plot_in=None):
    # group by label
    G_type_dic = {}
    for index, label in enumerate(labels):
        if label in (G_type_dic):
            G_type_dic[label] = G_type_dic[label] + [index]
        else:
            G_type_dic[label] = [index]

    # pre processing the recon and train set
    reconstructed_graphs = torch.stack(reconstructed_graphs)
    reconstructed_graphs = reconstructed_graphs.cpu().detach().numpy()
    reconstructed_graphs[reconstructed_graphs >= 0.5] = 1
    reconstructed_graphs[reconstructed_graphs < 0.5] = 0
    reconstructed_graphs = [nx.from_numpy_matrix(reconstructed_graphs[i]) for i in
                            range(reconstructed_graphs.shape[0])]
    reconstructed_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                            reconstructed_graphs if not nx.is_empty(G)]

    train_set = torch.stack(train_set)
    train_set = train_set.cpu().detach().numpy()
    train_set = [nx.from_numpy_matrix(train_set[i]) for i in range(train_set.shape[0])]
    train_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                 train_set if not nx.is_empty(G)]
    # graphPloter([G], graph_save_path + "ReconstructedRandomSampleAtEpoch_" + str(epoch))
    # mmd of each category
    log = ""
    for graph_type in G_type_dic.keys():

        print("mmd of " + str(graph_type) + " with " + str(len(G_type_dic[graph_type])) + " graphs")

        reconstr_log = mmd_eval([reconstructed_graphs[i] for i in G_type_dic[graph_type]],
                                [train_set[i] for i in G_type_dic[graph_type]], diam=True)
        logging.info(reconstr_log)
        log+=  str(graph_type)+"\n"+reconstr_log
        if plot_in!=None:
            # randomly selection of 10 samples
            r_graphs = G_type_dic[graph_type][:10]
            g_name = ''.join(e for e in graph_type if e.isalnum())
            for i in r_graphs:
                # plot reconstructed
                G  = reconstructed_graphs[i]
                # remove isolated nodes
                G.remove_edges_from(nx.selfloop_edges(G))
                G.remove_nodes_from(list(nx.isolates(G)))
                # plotter.plotG(G, "generated" + dataset, file_name=graph_save_path+"ReconstructedSample_At_epoch"+str(epoch))
                graphPloter([G], plot_in+ g_name + str(i) + "_ReconstructedRandomSample_")

                # plot origianl
                G = train_set[i]
                # remove isolated nodes
                G.remove_edges_from(nx.selfloop_edges(G))
                G.remove_nodes_from(list(nx.isolates(G)))
                # plotter.plotG(G, "generated" + dataset, file_name=graph_save_path+"ReconstructedSample_At_epoch"+str(epoch))
                graphPloter([G], plot_in+ g_name + str(i) + "_TargetRandomSample_" )
    if plot_in!=None:
        with open(plot_in+'_MMD_by_label.log', 'w') as f:
            f.write(log)

def resdul(target, perdic):
        return target - perdic

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, num_nodes, alpha, reconstructed_adj_logit, pos_wight, norm, node_num ,pca_transMtRX,pca_MeanMtRX,explained_variance_):
    device = reconstructed_kernel_val[0].device
    # loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight)

    norm =    mean.shape[0] * mean.shape[1]
    kl = (1/norm)* -0.5 * torch.sum(1+2*log_std - mean.pow(2)-torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum()/float(reconstructed_adj.shape[0]*reconstructed_adj.shape[1]*reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []
    kernels_residul = []
    # alpha = [1,1,10,10,1,1,3,3,3,3,3,3,1,1]
    for i in range(len(target_kernel_val)):
        if pca_transMtRX[i]!=None:

                reconstructed_kernel_val[i] = PCA_transform(reconstructed_kernel_val[i],pca_transMtRX[i],pca_MeanMtRX[i],explained_variance_[i])

        if single_variance:
            log_sigma = (torch.clamp(((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean(),min=0.00000000006)).sqrt().log()
        else:
            log_sigma = (torch.clamp(((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean(0),
                                 min=0.00000000006)).sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        kernels_residul.append(resdul(target_kernel_val[i], reconstructed_kernel_val[i]))

        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        each_kernel_loss.append(step_loss.cpu().detach().numpy()) #*alpha[i])
        kernel_diff += step_loss #*alpha[i]

    # kernel_diff += loss*alpha[-2]
    kernel_diff += kl * alpha[-1]
    # each_kernel_loss.append((loss*alpha[-2]).item())
    each_kernel_loss.append((kl * alpha[-1]).item())
    return kl , acc, kernel_diff, each_kernel_loss, kernels_residul

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


#graphPloter([nx.from_scipy_sparse_matrix(list_adj[6])],graph_save_path+"ReconstructedRandomSampleAtEpoch_")



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
        # list_label = None
        train_list_adj, test_list_adj, train_list_x , _,train_label,_ = data_split(list_adj,list_x,list_label)

        val_adj = train_list_adj[:int(len(test_list_adj)/2)]
        #---------------------------------------------------------------------------------------------------------
        list_adj = train_list_adj[int(len(val_adj)):]
        list_x = train_list_x[int(len(val_adj)):]
        # if list_label!=None:
        #     train_label = train_label[int(len(val_adj)):]
        if type(list_label)== list:
            list_label = train_label[int(len(val_adj)):]
        #--------------------------------------------------------------------------------------------------------
        list_graphs = Datasets(list_adj, self_for_none, list_x,list_label,Max_num=max_size, set_diag_of_isol_Zer=False)
    if task=="graphClasssification":
        list_graphs = Datasets(list_adj, self_for_none, list_x,list_label,Max_num=max_size)


print("#------------------------------------------------------")
fifty_fifty_dataset = list_adj + test_list_adj

fifty_fifty_dataset = [nx.from_numpy_matrix(graph.toarray()) for graph in fifty_fifty_dataset]
fifty_fifty_dataset = [ nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in fifty_fifty_dataset]

random.shuffle(fifty_fifty_dataset)

if ideal_Evalaution:
    print("some statistics of dataset")
    Diam_stats(fifty_fifty_dataset)
    print("50%50% Evalaution of dataset")
    logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset)/2)],fifty_fifty_dataset[int(len(fifty_fifty_dataset)/2):],diam=True))
    print("20%20% Evalaution of dataset")
    logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset) / 5)],
                          fifty_fifty_dataset[int(len(fifty_fifty_dataset) / 4):], diam=True))
print("#------------------------------------------------------")

del fifty_fifty_dataset
list_graphs.processALL(self_for_none=self_for_none)

adj_list_tmp = list_graphs.get_adj_list()
adj_list_tmp = [torch.tensor(adj_.todense()) for adj_ in adj_list_tmp]
adj_list_tmp = torch.stack(adj_list_tmp).float().to(device)
kernel_ = kernel(kernel_type = [""])
adj_list_tmp = adj_list_tmp * (1 - torch.eye(adj_list_tmp.shape[-1], adj_list_tmp.shape[-1])).to(device)

k_steps = kernel_.S_randomWalks(adj_list_tmp, max_needed_step)

SubGraphNodeNum = subgraphSize if subgraphSize!=None else list_graphs.max_num_nodes
in_feature_dim = list_graphs.feature_size # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes


# defining the histograms:

binners = []
for _binner in binningStrategy:
    if _binner=="EqualWidth":
        binners.append(Binner.EqualWidthBinner)

    elif _binner=="UniformBinner":
        binners.append(Binner.UniformBinner)

    elif _binner=="EqualFreq":
        binners.append(Binner.EqualFreqBinner)

    elif _binner=="BounderBinner":
        binners.append(Binner.BounderBinner)
if bin_num == "sqrt(N)" or bin_num == "sqrt_N":
    num_of_bin = int(np.sqrt(list_graphs.max_num_nodes))
elif bin_num== "N":
    num_of_bin = int((list_graphs.max_num_nodes))

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
_Binner = Binner()
degree_center, degree_width = _Binner.combineBinner(binners,adj_list_tmp.sum(-1), num_of_bin, UpperBound = SubGraphNodeNum)
degree_hist = Histogram((degree_center).to(device), degree_width.to(device), MembershipFunction)
#--------------------------
bin_center = torch.tensor([[i/SubGraphNodeNum] for i in range(0, SubGraphNodeNum+1)]).to(device)
bin_width = torch.tensor([[.1] for i in bin_center]).to(device)
binwith =(1/SubGraphNodeNum)

tri =    (torch.matmul(adj_list_tmp, torch.matmul(adj_list_tmp, adj_list_tmp)))
tri = torch.diagonal(tri, dim1=1, dim2=2)
degree = adj_list_tmp.sum(-1)
all_possibles = degree * (degree - 1)+.00000001

# # all_possibles = torch.clamp(all_possibles, min=.001)
# mask = all_possibles<=0
# mask = torch.tensor(mask)+0
# all_possibles = all_possibles + mask
clustering_coef = (tri / all_possibles)

center, width = _Binner.combineBinner(binners, clustering_coef, num_of_bin, UpperBound = 1)
clusteringCoefficientHist = Histogram(center, width, MembershipFunction)
#--------------------------
HistogramOfRandomWalks = []
for i_ in range(step_num):
    i = i_+1
    bin_center, binwith = _Binner.combineBinner(binners, k_steps[i_], num_of_bin, UpperBound=SubGraphNodeNum**i )
    HistogramOfRandomWalks.append(Histogram(bin_center.to(device),binwith.to(device),MembershipFunction))
#--------------------------
# cycles Histogram
cyclesHistogram = []
for cycle_ in range(max_size_of_cyc-2):
    center, binwith = _Binner.combineBinner(binners, torch.diagonal(k_steps[cycle_+1], dim1=1, dim2=2), num_of_bin, UpperBound = SubGraphNodeNum**(cycle_+2))
    cyclesHistogram.append(Histogram(center.to(device),binwith.to(device),MembershipFunction))
#--------------------------


reach_hist = []
Adj_k_th = adj_list_tmp
for i_ in range(steps_of_reachability):
    i = i_ + 1
    Adj_k_th = torch.matmul(adj_list_tmp, Adj_k_th)
    Adj_k_th = torch.clamp(Adj_k_th, max=1)
    bin_center, binwith = _Binner.combineBinner(binners, Adj_k_th.sum(-1), num_of_bin, UpperBound=SubGraphNodeNum )
    reach_hist.append(Histogram( bin_center.to(device), binwith.to(device),MembershipFunction))
# --------------------------

del k_steps, adj_list_tmp, kernel_
kernel_model = kernel(cyclesHistogram=cyclesHistogram, kernel_type = kernl_type, step_num = step_num, steps_of_reachability = steps_of_reachability,degree_hist = degree_hist,
                      ReachabilityInKsteps=reach_hist, max_size_of_cyc = max_size_of_cyc, HistogramOfRandomWalks = HistogramOfRandomWalks, clusteringCoefficientHist=clusteringCoefficientHist)
#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

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
num_itr  = epoch_number*max(int(len(list_graphs.list_adjs)/mini_batch_size),1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_itr, pct_start=.25)


# pos_wight = torch.true_divide((list_graphs.max_num_nodes**2*len(list_graphs.processed_adjs)-list_graphs.toatl_num_of_edges),
#                               list_graphs.toatl_num_of_edges) # addrressing imbalance data problem: ratio between positve to negative instance
# pos_wight = torch.tensor(40.0)
# pos_wight/=10
num_nodes = list_graphs.max_num_nodes
#ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)


start = timeit.default_timer()
# Parameters
step =0
swith = False
print(model)
logging.info(model.__str__())
min_loss = float('inf')

list_graphs.shuffle()
if(subgraphSize==None):
    list_graphs.processALL(self_for_none = self_for_none)
    adj_list = list_graphs.get_adj_list()
    graphFeatures , _ = get_subGraph_features(adj_list, None, kernel_model)
    # PCA_Analiz(graphFeatures[2])
    whiten = False
    pca_MeanMtRX = [None for x in graphFeatures]
    pca_transMtRX = [None for x in graphFeatures]
    pca_VAr = [None for x in graphFeatures]
    if PCATransformer:
        for f, feature in enumerate(graphFeatures):
            print(functions[f + 2])
            if feature.shape[-1]>3 and len(feature.shape)>1:

                feature = graphFeatures[f].cpu().detach().numpy()
                if  rand_strategy:
                    _tmp = PCA_projector(feature, whiten=whiten).components_.T

                    pca_transMtRX[f] =torch.rand((_tmp.shape[0],_tmp.shape[1]))
                else:
                #     graphFeatures[f] = PCA_transform(feature,torch.randn_like(pca_transMtRX[f]), None,None)
                    pca,scalar = PCA_projector(feature,whiten=whiten, normalize=scale)

                    pca_transMtRX[f] = torch.tensor(pca.components_[pca.explained_variance_ratio_>0.001].T,dtype=torch.float32,device="cpu")
                    pca_MeanMtRX[f] = torch.tensor(pca.mean_, dtype=torch.float32, device="cpu")

                #
                    if  scale and type(scalar.var_)==numpy.ndarray:
                        pca_VAr[f] =torch.tensor(scalar.var_, dtype=torch.float32, device="cpu")




                feature = torch.tensor(feature,dtype =torch.float32)
                # if not rand_strategy:
                graphFeatures[f] = PCA_transform(feature,pca_transMtRX[f], pca_MeanMtRX[f],pca_VAr[f])
                # else:
                #     graphFeatures[f] = PCA_transform(feature,torch.randn_like(pca_transMtRX[f]), None,None)
                        # import pandas as pd
                        # import matplotlib.pyplot as plt
                        # import matplotlib.pyplot as plt
                        #
                        # plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
                        #
                        # # Plot Histogram on x
                        # x = graphFeatures[1][:, 3].detach().cpu().numpy().reshape(-1)
                        # plt.hist(x, bins=50)
                        # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
                        # plt.show()

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

invarients_residual_hist = []

#-------------------------------------------------------------
# cal diam
graphs = [nx.from_scipy_sparse_matrix(g) for g in list_graphs.list_adjs]
graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in graphs]
diam = [nx.diameter(G) for G in graphs]
print("the train set average Diam(Only the max Con Component is considered): ", diam)
lrs = []
#-------------------------------------------------------------
for epoch in range(epoch_number):
    if epoch==14:
        print()
    list_graphs.shuffle()
    batch = 0
    lbl = []
    rec_graphs = []
    graphs_in_this_epoch = []
    for iter in range(0, max(int(len(list_graphs.list_adjs)/mini_batch_size),1)*mini_batch_size, mini_batch_size):
        from_ = iter
        to_= mini_batch_size*(batch+1)
    # for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
    #     from_ = iter
    #     to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)

        if subgraphSize==None:
            org_adj,x_s, node_num, subgraphs_indexes, target_kelrnel_val,lables = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)
        else:
            org_adj,x_s, node_num, subgraphs_indexes,lables = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)

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
        kl_loss, acc, kernel_cost,each_kernel_loss,dim_aspect = OptimizerVAE(reconstructed_adj, generated_kernel_val, subgraphs.to(device), [val.to(device) for val in target_kelrnel_val]  , post_log_std, post_mean, num_nodes, alpha,reconstructed_adj_logit, pos_wight, 2,node_num,pca_transMtRX,pca_MeanMtRX,pca_VAr)
        if lables!=None:
            lbl += lables
            rec_graphs += reconstructed_adj
            graphs_in_this_epoch+=subgraphs

            # train_set = subgraphs.cpu().detach().numpy()
            # train_set = [nx.from_numpy_matrix(train_set[i]) for i in range(train_set.shape[0])]
            # train_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
            #              train_set if not nx.is_empty(G)]
            # graphPloter([train_set[10]], graph_save_path + "ReconstructedRandomSampleAtEpoch_" + str(epoch))

            # cal diam
            # graphs = [graph.cpu().detach().numpy() for graph in graphs_in_this_epoch]
            # graphs = [nx.from_numpy_matrix(g) for g in graphs]
            # graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in graphs]
            # diam = [nx.diameter(G) for G in graphs]
        if ResidualAnlyze:
            invarients_residual_hist.append(dim_aspect) #recording the residual history
            # feature_i = 0
            # feature = [log[feature_i].detach().cpu().numpy() for log in invarients_residual_hist]
            # feature = np.concatenate(feature, axis=0)
            #
            # co0 = np.corrcoef(feature.transpose())
            # numpy.savetxt(functions[feature_i + 2] + ".csv", co0, delimiter=",")

        loss = kernel_cost


        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), *each_kernel_loss],tmp, redraw= redraw)  # ["Accuracy", "loss", "AUC"])

        step+=1
        optimizer.zero_grad()
        lrs.append(optimizer.param_groups[0]["lr"])
        loss.backward()

        if keepThebest and min_loss>loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()
        scheduler.step()


        if (step+1) % visulizer_step == 0 or epoch == (epoch_number)-1:

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
            #remove isolated nodes
            G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            # plotter.plotG(G, "generated" + dataset, file_name=graph_save_path+"ReconstructedSample_At_epoch"+str(epoch))
            graphPloter([G],graph_save_path+"ReconstructedRandomSampleAtEpoch_"+str(epoch))
            #---------------------------------------------------------------------------
            print("RECONSTRUCTED graph vs Input (all connected component):")
            logging.info("RECONSTRUCTED graph vs Input (all connected component):")

            reconstructed_adj = reconstructed_adj.cpu().detach().numpy()
            reconstructed_adj[reconstructed_adj >= 0.5] = 1
            reconstructed_adj[reconstructed_adj < 0.5] = 0
            reconstructed_adj = [nx.from_numpy_matrix(reconstructed_adj[i]) for i in range(reconstructed_adj.shape[0])]
            reconstructed_adj = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                 reconstructed_adj if not nx.is_empty(G)]

            target_set = subgraphs.cpu().detach().numpy()
            target_set = [nx.from_numpy_matrix(target_set[i]) for i in range(target_set.shape[0])]
            target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                 target_set if not nx.is_empty(G)]


            reconstr_log = mmd_eval(reconstructed_adj, target_set, diam=True)
            logging.info(reconstr_log)

            # ---------------------------------------------------------------------------

            if task=="graphGeneration":
                #comparing generated graphs and Validation set
                print("-------------------------------------------------------")
                print("comparision with VALIDATION set:")
                logging.info("comparision with VALIDATION set:")
                EvalTwoSet(model, val_adj, graph_save_path, Save_generated= False,_f_name=epoch)
                print("-------------------------------------------------------")
                # mmd for each category/label
                if len(rec_graphs) > 0:
                    mmdBylabel(rec_graphs, graphs_in_this_epoch, lbl)

                # if ((step+1) % visulizer_step*2):
                #     torch.save(model.state_dict(), graph_save_path+"model_"+str(epoch)+"_"+str(batch))
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



print(lrs)
model.eval()
torch.save(model.state_dict(), graph_save_path+"model_"+str(epoch)+"_"+str(batch))

stop = timeit.default_timer()
print("trainning time:", str(stop-start))
logging.info("trainning time: "+ str(stop-start))
# save the train loss for comparing the convergencenan
import json

file_name = graph_save_path+"_"+encoder_type+"_"+decoder_type+"_"+dataset+"_"+task+"_"+args.model+"_elbo_loss.txt"

# with open(file_name, "w") as fp:
#     json.dump(list(np.array(pltr.values_train[-2])+np.array(pltr.values_train[-1])), fp)
#
#
# with open(file_name+"_CrossEntropyLoss.txt", "w") as fp:
#     json.dump(list(np.array(pltr.values_train[-2])), fp)
#
#
# with open(file_name+"_train_loss.txt", "w") as fp:
#     json.dump(pltr.values_train[1], fp)


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

print("-------------------------------------------------------")
print("comparision with VALIDATION set:")
EvalTwoSet(model, val_adj, graph_save_path+"_Val_", Save_generated=True, _f_name=epoch)
if task=="graphGeneration" :
    print("-------------------------------------------------------")
    print("comparision with VALIDATION set:")
    EvalTwoSet(model, val_adj, graph_save_path+"Val_", Save_generated=True, _f_name=epoch)
    if EvalOnTest==True:
        print("comparision with TEST set:")
        EvalTwoSet(model, test_list_adj, graph_save_path+"Test_", Save_generated= True,_f_name="final_eval", OnlyMax_Con_com=False)
    print("-------------------------------------------------------")
    if write_them_in!=None:
        import os
        #  create the directory if it does not exist
        if not os.path.exists(write_them_in):
            os.makedirs(write_them_in)

        with open(write_them_in+'configs.log', 'w') as f:
            conf = str(args)+"\nalpha: " + str(alpha) + "   max_size_of_cyc,steps_of_reachability, step_num: " + str(
            [max_size_of_cyc, steps_of_reachability, step_num])+ "  binningStrategy: "+ str(binningStrategy)
            f.write(conf)


        pltr.save_plot(write_them_in + "KernelVGAE_log_plot")
        log = "Reconstructed: \n  maximum_connected_componnent: "+ reconstr_log

        log += "\nVal: \n" + EvalTwoSet(model, val_adj, write_them_in + "_Gen_for_Val_", Save_generated=True,
                                               _f_name="final_eval")

        if EvalOnTest:
            log +="\n Generated: \n" + EvalTwoSet(model, test_list_adj, write_them_in + "_Gen_for_Test_", Save_generated=True,
                                             _f_name="final_eval")

        with open(write_them_in+'MMD.log', 'w') as f:
            f.write(log)

        # drop the reconstructed graph in the last epoch
        reconstructed_adj = [nx.to_numpy_array(G) for G in reconstructed_adj]
        np.save(write_them_in + 'reconstructed_graph.npy', reconstructed_adj,
                    allow_pickle=True)
        recon_target_set = [nx.to_numpy_array(G) for G in target_set]
        np.save(write_them_in + 'reconstructed_graph_target.npy', recon_target_set,
                allow_pickle=True)

        # drop the validation graph in the last epoch
        val_adj = [np.array(G) for G in val_adj]
        np.save(write_them_in + 'Val_graph.npy', target_set,
                allow_pickle=True)

        # drop the test graph in the last epoch
        if EvalOnTest == True:
            target_set = [nx.to_numpy_array(G) for G in test_list_adj]
            np.save(write_them_in + 'Test_graph.npy', target_set,
                allow_pickle=True)

        # drop generated graphs
        if EvalOnTest == True:
            generated_graphs = graph_generator( model, mini_batch_size)
            target_set = [nx.to_numpy_array(G) for G in generated_graphs]
            np.save(write_them_in + 'Generated_graphs.npy', target_set,
                    allow_pickle=True)

        if type(lbl)==list and len(lbl)>0:
            mmdBylabel(rec_graphs, graphs_in_this_epoch, lbl,write_them_in)
