
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

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


subgraphSize = None
keepThebest = False

parser = argparse.ArgumentParser(description='Kernel VGAE')


parser.add_argument('-e', dest="epoch_number" , default=40000, help="Number of Epochs", type=int)
parser.add_argument('-v', dest="Vis_step", default=4000000, help="model learning rate")
# parser.add_argument('-AE', dest="AutoEncoder", default=True, help="either update the log plot each step")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")

parser.add_argument('-lr', dest="lr", default=0.0003, help="model learning rate",type=float) # henerally   0.0003 works well # 0005 for pca+ diagonal varianve
parser.add_argument('-dataset', dest="dataset", default="lobater", help="possible choices are:  mnist_01_percent,random_powerlaw_tree, PVGAErandomGraphs_dev, PVGAErandomGraphs, IMDBBINARY, IMDbMulti,wheel_graph, star, triangular_grid, multi_community, NCI1, ogbg-molbbbp, IMDbMulti, grid, community, citeseer, lobster, DD")#citeceer: ego; DD:protein


parser.add_argument('-NofCom', dest="num_of_comunities", default=128, help="Number of comunites",type=int)
parser.add_argument('-graphEmDim', dest="graphEmDim", default=128, help="the simention of graphEmbedingLAyer",type=int)
parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
parser.add_argument('-graph_save_path', dest="graph_save_path", default=None, help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature" , default=True, help="either use features or identity matrix")
parser.add_argument('-PATH', dest="PATH" , default="model", help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder" , default="FC_R", help="the decoder type, FC or SBM")
parser.add_argument('-encoder', dest="encoder_type" , default="AvePool", help="the encoder")    #"diffPool" "AvePool" "GCNPool"
parser.add_argument('-batchSize', dest="batchSize" , default=64, help="the size of each batch")
parser.add_argument('-UseGPU', dest="UseGPU" , default=True, help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model" , default="PMI", help=" PMI")
parser.add_argument('-device', dest="device" , default="cuda:0", help="Which device should be used")
parser.add_argument('-task', dest="task" , default="graphGeneration", help="linkPrediction, GraphCompletion, graphClasssification or graphGeneration")
parser.add_argument('-BFS', dest="bfsOrdering" , default=True, help="use bfs for graph permutations", type=bool)
parser.add_argument('-directed', dest="directed" , default=False, help="is the dataset directed?!", type=bool)
parser.add_argument('-beta', dest="beta" , default=1, help="beta coefiicieny", type=float)
parser.add_argument('-limited_to', dest="limited_to" , default=-1, help="How many instance you want to pick, its for reducing trainning time in dev", type=int)
parser.add_argument('-ideal_Evalaution', dest="ideal_Evalaution" , default=False, help="if you want to comapre the 50%50 subset of dataset comparision?!", type=bool)
parser.add_argument("-desc_fun", dest="desc_fun" , default= [ "NumberOfVer","in_degree_dist", "HistogramOfCycles","HistogramOfRandomWalks","ReachabilityInKsteps"], help="the descriptor function which should be used: ", nargs='+',)#[ "in_degree_dist", "HistogramOfCycles","HistogramOfRandomWalks","ReachabilityInKsteps"]
# kernl_type = ["TotalNumberOfTriangles", "HistogramOfRandomWalks","in_degree_dist", "out_degree_dist", "ReachabilityInKsteps","HistogramOfCycles"] # "clusteringCoefficient", "cycles_number","ReachabilityInKsteps" ReachabilityInKsteps "steps_of_reachability"
parser.add_argument('-write_them_in', dest="write_them_in" , default=None, help="path_to write generated/test graphs" )
parser.add_argument('-PCATransformer', dest="PCATransformer" , default=False, help="either use PCA transformer or net", )
parser.add_argument('-rand_strategy', dest="rand_strategy" , default=False, help="either use rand matrix instead of  PCA componnents", )
parser.add_argument('-ResidualAnlyze', dest="ResidualAnlyze" , default=False, help="either save the residuls for feuture znzlyze or not", )
parser.add_argument('-EvalOnTest', dest="EvalOnTest" , default=True, help="either print the statistics based EVAL on the test set", type=bool)
parser.add_argument('-binningStrategy', dest="binningStrategy" , default=["EqualWidth","EqualFreq","BounderBinner"], help="binning schema: UniformBinner, EqualWidth or EqualFreq", nargs='+' )
parser.add_argument('-single_variance', dest="single_variance" , default=True, help="use a shared variancee for each invarients, othervise use a variance for each dim", )
parser.add_argument('-membershipFunction', dest="membershipFunction" , default="gaussian", help="soft membership function; either tri(triangle) or gaussian", )
parser.add_argument('-scale', dest="scale" , default=False, help="either scale an invaient dimenssion or not as preprosseing of PCA", )
parser.add_argument('-bin_num', dest="bin_num" , default="sqrt(N)", help="number of bins: eitehr sqrt(N) or N", )
parser.add_argument('-arg_step_num"', dest="arg_step_num" , default=3, help="Number of Epochs", type=int)
parser.add_argument('-steps_of_reachability', dest="arg_steps_of_reachability" , default=3, help="Number of Epochs", type=int)
parser.add_argument('-max_size_of_cyc', dest="arg_max_size_of_cyc" , default=3, help="max graph cycle in the cycle kernel", type=int)
parser.add_argument('-scheduler_type', dest="scheduler_type" , default="OneCyle", help="OneCyle or CyclicLR, OR None", )
parser.add_argument('-Model_seed', dest="Model_seed" , default=0, help="the seed for initializing the lr", type=int)
parser.add_argument('-Dataset_seed', dest="Dataset_seed" , default=0, help="the seed for initializing the numpy; affects the dataset slit", type=int)
parser.add_argument('-in_normed_layer', dest="in_normed_layer" , default=True, help="either normalize the inpput ot not", type=bool)
parser.add_argument('-data_Normalizer', dest="data_Normalizer" , default=False, help="either normalize the inpput ot not", type=bool)
parser.add_argument('-verbose', dest="verbose" , default=False, help="either save the track of model in trainning", type=bool)
parser.add_argument('-TypeOFnormLayers', dest="normLayers" , default="LayerNorm", help="the type of norm layer following the FCNs",)
parser.add_argument('-LDP', dest="LDP" , action="store_true", help="this paprameter indicate if the graph generation should be under LDP, if LDP == True then graph statisics would be in_degree_dist and HistogramOfCycles, we have not implemented the  Random mechanism for the other statistics ")
parser.add_argument("-epsilon",dest="epsilon" , default=4,  help="The LDP parameter",type=float )
parser.add_argument("-NumDecLayers",dest="NumDecLayers" , default=4,  help="The number of layers in the decoder",type=int )
parser.add_argument("-reach_act",dest="reach_act" , default="leaky",  help="the activation funtion of  reachability kernel")
parser.add_argument("-ignore_cycles",dest="ignore_cycles" , action="store_false",  help="either ignore diagonal entries (counting less loops) in the random walks calculation")

# parser.add_argument('-TypeOFnormLayers', dest="normLayers" , default="LayerNorm", help="the type of norm layer following the FCNs",)
args = parser.parse_args()
reach_act= args.reach_act
ignore_cycles = args.ignore_cycles
NumDecLayers = args.NumDecLayers
LDP = args.LDP

epsilon = args.epsilon
# torch.autograd.set_detect_anomaly(True)
verbose = args.verbose
normLayers = args.normLayers
in_normed_layer = args.in_normed_layer
data_Normalizer = args.data_Normalizer
scheduler_type = args.scheduler_type
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
kernl_type = args.desc_fun
arg_step_num= args.arg_step_num
arg_steps_of_reachability= args.arg_steps_of_reachability
arg_max_size_of_cyc= args.arg_max_size_of_cyc
seeds = args.Model_seed
np_seed = args.Model_seed
torch.manual_seed(seeds)
torch.cuda.manual_seed(seeds)
torch.cuda.manual_seed_all(seeds)
np.random.seed(np_seed)
random.seed(np_seed)

if True == args.directed:
    raise Exception("Sorry, This implemetation assumes the graph are undirected and simple")

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

#===============================================================================================
# just ot append ex
import sys
def catchException(logger, typ, value, traceback):
    logger.critical("My Error Information")
    logger.critical("Type: %s" % typ)
    logger.critical("Value: %s" % value)
    logger.critical("Traceback: %s" % traceback)

# Use a partially applied function
func = lambda typ, value, traceback: catchException(logging, typ, value, traceback)
sys.excepthook = func
#===============================================================================================

# **********************************************************************
# setting
print("KernelVGAE SETING: "+str(args))
logging.info("KernelVGAE SETING: "+str(args))
PATH = args.PATH # the dir to save the with the best performance on validation data


alpha = [ 1]
# the steps number are hardcoded, it can also be hyperparameter
max_size_of_cyc = 3
steps_of_reachability =2
step_num = 2
if arg_max_size_of_cyc!=None:
    max_size_of_cyc = arg_max_size_of_cyc
if arg_steps_of_reachability != None:
    steps_of_reachability = arg_steps_of_reachability
if arg_step_num != None:
    step_num = arg_step_num


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

if LDP == True :
    # random mechanism has implemented only for this statistics, the implementation can be exteded to other statistics
    kernl_type=[ "in_degree_dist","HistogramOfCycles"]
    max_size_of_cyc = 3
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
functions= ["loss"]
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



pltr = plotter.Plotter(save_to_filepath="MMD_kernelVGAE_Log",functions=functions)






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



class FC_Encoder(torch.nn.Module):
    def __init__(self, in_feature_dim, hiddenLayers = [256, 256, 256], GraphLatntDim=1024, dataNormalizer=False, TypeOFnormLayers = "LayerNorm"):
        super(FC_Encoder, self).__init__()
        self.dataNormalizer = dataNormalizer
        self.dataNomalizerLayer = torch.nn.BatchNorm1d(in_feature_dim,affine=None, momentum=None)

        hiddenLayers = [in_feature_dim]+hiddenLayers + [GraphLatntDim]
        if TypeOFnormLayers== "LayerNorm":
            self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(hiddenLayers[i+1],elementwise_affine=False) for i in range(len(hiddenLayers) - 1)])
        else:
            self.normLayers = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(hiddenLayers[i + 1], affine=None, momentum=None) for i in
                 range(len(hiddenLayers) - 1)])
        # self.normLayers.append(torch.nn.LayerNorm(hiddenLayers[-1],elementwise_affine=False))
        # self.normLayers = torch.nn.ModuleList(
        #     [torch.nn.BatchNorm1d(hiddenLayers[i + 1]) for i in range(len(hiddenLayers) - 1)])
        # self.normLayers.append(torch.nn.BatchNorm1d(GraphLatntDim))
        self.encoder_layers = torch.nn.ModuleList([torch.nn.Linear(hiddenLayers[i], hiddenLayers[i + 1], bias=True) for i in range(len(hiddenLayers) - 1)])

        self.stochastic_mean_layer = torch.nn.Linear(GraphLatntDim, GraphLatntDim)
        self.stochastic_log_std_layer = torch.nn.Linear(GraphLatntDim, GraphLatntDim)


    def forward(self, invarients, activation= torch.nn.LeakyReLU(0.01)):
        h= invarients
        if self.dataNormalizer:
            h = self.dataNomalizerLayer(h)

        for i in range(len(self.encoder_layers)):
            h= self.encoder_layers[i](h)
            h= activation(h)
            # if((i<len(self.GCNlayers)-1)):
            h = self.normLayers[i](h)

        mean = self.stochastic_mean_layer(h)
        log_std = self.stochastic_log_std_layer(h)

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

    def forward(self, invarients):
        """
        :param graph: normalized adjacency matrix of graph
        :param features: normalized node feature matrix
        :return:
        """
        mean, log_std = self.encode( invarients)
        samples = self.reparameterize(mean, log_std)
        reconstructed_adj_logit = self.decode(samples)
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)

        kernel_value = self.kernel(reconstructed_adj)
        return reconstructed_adj, samples, mean, log_std, kernel_value, reconstructed_adj_logit

    def reparameterize(self, mean, log_std):
        if self.AutoEncoder == True:
            return mean
        # var = torch.exp(log_std).pow(2)
        var = torch.exp(log_std)
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
    def calculate_difference(x,min_pooling=False):
        y = np.empty_like(x)
        size = len(x)
        if min_pooling==True:
            for i in range(size):
                if i == 0:
                    y[i] = x[i + 1] - x[i]
                elif i == size - 1:
                    y[i] = (x[i] - x[i - 1])
                else:
                    y[i] = min((x[i + 1] - x[i]),(x[i] - x[i - 1])) / 2
            return y
        for i in range(size):
            if i == 0:
                y[i] = x[i + 1] - x[i]
            elif i == size - 1:
                y[i] = (x[i] - x[i - 1])
            else:
                y[i] = (x[i + 1] - x[i - 1])/2

        return y

    def EqualWidthBinner(featureValues, numBins, minZero=True, UpperBound = None, minWidth=1 ):
        featureValues = featureValues.reshape(-1)#numpy.unique(featureValues.detach().cpu().numpy())
        min_ = 0
        if  not minZero:
            if len(featureValues[featureValues>0])<1:
                bin_width = [torch.tensor(minWidth).to(featureValues.device)]
                bin_center = [torch.tensor(0).to(featureValues.device)]
                return torch.tensor(bin_center), torch.tensor(bin_width)
            else:
                min_ = torch.min(featureValues[featureValues>0])
        # if UpperBound==None:
        #     max_ = torch.max(featureValues).detach().cpu().item()
        # else:
        #     max_= UpperBound
        max_ = torch.max(featureValues).detach().cpu().item()



        step = max(((max_ - min_) / numBins),minWidth)

        bin_width = [torch.tensor(step).to(featureValues.device)]
        bin_center = [torch.tensor(0).to(featureValues.device)]
        numBins = int(min(numBins,(max_ - min_)/minWidth))
        # if step != 0:
        for i in range(numBins):
            bin_center += [min_ + step * (i+1)]
            bin_width += [step]
        # else:
        #     bin_center += [min_]
        #     bin_width += [torch.tensor(1)]

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


    def EqualFreqBinner(featureValues, numBins, UpperBound=None,minWidth=.5):

        featureValues = featureValues.reshape(-1)

        bin_width = []
        bin_center = []
        # pre_binCenter = 0
        domain = np.unique(featureValues.detach().cpu().numpy())
        if len(domain)<numBins:
            # return torch.tensor(domain), torch.tensor([2. for x in (domain)])
            if len(domain)==1:
                width = torch.tensor([minWidth for x in (domain)])
            else: width = Binner.calculate_difference(domain)
            slope = torch.tensor([2. for x in range(0, len(width), 1)])
            width = torch.tensor(width) / slope
            return torch.tensor(domain),width

        featureValues = featureValues[featureValues > 0]
        featureValues = np.sort(featureValues.detach().cpu().numpy())
        max_ = np.max(featureValues)
        # numBins = min(len(np.unique(featureValues)),numBins)

        bin_center += [0]
        bin_width += [minWidth]

        b = 0
        numElementsInEachBin = int(len(featureValues) / (numBins))
        while b <(numBins):

            bin = featureValues if b == (numBins - 1) else featureValues[:numElementsInEachBin]

            binCenter = np.mean(bin)
            binWidth = np.std(bin)

            if (len(bin_center)==0 or binCenter>=bin_center[-1]+2*minWidth):
                bin_center += [binCenter]

                bin_width += [3*binWidth]
                b += 1
            featureValues = featureValues[numElementsInEachBin:]
            if (numBins-b)>0:
                numElementsInEachBin = max(int(len(featureValues) / (numBins - b)), 1)
            if len(featureValues)==0:
                break

        if (bin_center[-1]+2*minWidth)<max_:
            bin_center.append(max_)
        #----------------------------
        # slope = torch.tensor([1. for x in range(0, len(bin_center), 1)])
        # bin_width = torch.tensor(bin_width)/slope
        # bin_width[bin_width<minWidth] = minWidth
        # ----------------------------
        bin_width = Binner.calculate_difference(bin_center)
        slope = torch.tensor([2. for x in range(0, len(bin_width), 1)])
        bin_width = torch.tensor(bin_width) / slope
        bin_width[bin_width<minWidth] = minWidth
        bin_width[0] = minWidth
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

    def forward(self, vec, MAxBin=None,MinBin=None): # id guassian==False it will be triangle membership fun

        if self.MemberFun=="gaussian":
            score_vec = (vec.view(vec.shape[0], 1, vec.shape[-1]) - self.bin_center) / self.bin_width
            score_vec = torch.square(score_vec)*-1.
            score_vec = torch.exp(score_vec)
            hist_ =  score_vec.sum(-1)
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

            hist_ =  score_vec.sum(-1)
        else:
            raise Exception("Sorry, Membership function is not defined")
        if MAxBin != None:
           Out_of_Domain  = (vec*(torch.tensor(vec>MAxBin)+0.)).sum(-1,keepdim=True)
           hist_ = torch.cat((hist_,Out_of_Domain),dim=1)
        # if MinBin!= None:
        #     Out_of_Domain = ((torch.tensor(vec >0) + 0.)*vec * (torch.tensor(vec < MinBin) + 0.)).sum(-1, keepdim=True)
        #
        #     hist_ = torch.cat((hist_, Out_of_Domain,vec.sum((-1),keepdim=True)), dim=1)
        return  hist_
class GraphTransformerDecoder(torch.nn.Module):
    def __init__(self,input,lambdaDim,SubGraphNodeNum, layers= [128 ], normLayers = "LayerNorm"):
        super(GraphTransformerDecoder, self).__init__()
        self.lamda =torch.nn.Parameter(torch.Tensor(lambdaDim, lambdaDim))
        self.SubGraphNodeNum = SubGraphNodeNum
        layers = [input] + [2048, 1024]+[lambdaDim*SubGraphNodeNum]
        if normLayers=="LayerNorm":
            self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i+1],elementwise_affine=False) for i in range(len(layers) - 2)])
        else:
            self.normLayers = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(layers[i + 1], elementwise_affine=False) for i in range(len(layers) - 2)])

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

class GraphTransformerDecoder_FC_d(torch.nn.Module):
    def __init__(self,input,Max_degree,SubGraphNodeNum, TypeOFnormLayers = "LayerNorm"):
        super(GraphTransformerDecoder_FC_d, self).__init__()
        self.SubGraphNodeNum = SubGraphNodeNum
        layers = [input] + [1024,2048, 1024]
        self.Max_degree = Max_degree
        layers = layers + [int(Max_degree*SubGraphNodeNum)]
        if TypeOFnormLayers=="LayerNorm":
            self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i+1],elementwise_affine=False) for i in range(len(layers) - 2)])
        else:
            self.normLayers = torch.nn.ModuleList([torch.nn.BatchNorm1d(layers[i+1],affine=None, momentum=None) for i in range(len(layers) - 2)])


        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1],torch.float32) for i in range(len(layers) - 1)])

    def forward(self, in_tensor, subgraphs_indexes=None, activation= torch.nn.LeakyReLU(0.001)):

        for i in range(len(self.layers)):
            # in_tensor = self.normLayers[i](in_tensor)
            in_tensor = self.layers[i](in_tensor)
            if i !=len((self.layers))-1:
                in_tensor = activation(in_tensor)
                in_tensor = self.normLayers[i](in_tensor)
        else:
            zero = -200
            ADJ = torch.zeros((in_tensor.shape[0],SubGraphNodeNum,SubGraphNodeNum),device=in_tensor.device)+zero
            # in_tensor = in_tensor.reshape(in_tensor.shape[0],-1)
            index = torch.tensor(torch.arange(self.SubGraphNodeNum).reshape(-1, 1).repeat(1, self.Max_degree))
            index_0 = index - torch.flip(torch.arange(self.Max_degree), [0])-1
            # ADJ[:,:,torch.tril_indices(SubGraphNodeNum,SubGraphNodeNum,-1)[1]] = in_tensor
            Zeroes = (index_0.reshape(-1)<0).nonzero().reshape(-1)
            ADJ[ :,index.reshape(-1), index_0.reshape(-1)] = in_tensor
            ADJ[:, index.reshape(-1)[Zeroes], index_0.reshape(-1)[Zeroes]] = zero
            #transpose
            ADJ[ :,index_0.reshape(-1),index.reshape(-1), ] = in_tensor
            ADJ[:, index_0.reshape(-1)[Zeroes], index.reshape(-1)[Zeroes]] = zero

            # Diagonal entries
            ADJ[:, list(range(SubGraphNodeNum)), list(range(SubGraphNodeNum))] = zero
            # ADJ = ADJ + ADJ.permute(0,2,1)



        return ADJ


class GraphTransformerDecoder_FC(torch.nn.Module):
    def __init__(self,input,LargestGraphSize, TypeOFnormLayers = "LayerNorm"):
        super(GraphTransformerDecoder_FC, self).__init__()
        self.LargestGraphSize = LargestGraphSize

        layers = [input] + [4056,2048,4056, 2048]


        layers = layers + [int((LargestGraphSize**2 - LargestGraphSize)/2)]
        if TypeOFnormLayers=="LayerNorm":
            self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i+1],elementwise_affine=False) for i in range(len(layers) - 2)])
        else:
            self.normLayers = torch.nn.ModuleList([torch.nn.BatchNorm1d(layers[i+1],affine=None, momentum=None) for i in range(len(layers) - 2)])

        # self.normLayers = torch.nn.ModuleList(
        #     [torch.nn.BatchNorm1d(layers[i + 1], ) for i in range(len(layers) - 2)])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1],torch.float32) for i in range(len(layers) - 1)])

    def forward(self, in_tensor, activation= torch.nn.LeakyReLU(0.001)):

        for i in range(len(self.layers)):
            # in_tensor = self.normLayers[i](in_tensor)
            in_tensor = self.layers[i](in_tensor)
            if i !=len((self.layers))-1:
                in_tensor = activation(in_tensor)
                in_tensor = self.normLayers[i](in_tensor)


        ADJ = torch.zeros((in_tensor.shape[0],self.LargestGraphSize,self.LargestGraphSize),device=in_tensor.device)
        ADJ[:,torch.tril_indices(self.LargestGraphSize,self.LargestGraphSize,-1)[0],torch.tril_indices(self.LargestGraphSize,self.LargestGraphSize,-1)[1]] = in_tensor
        ADJ = ADJ + ADJ.permute(0,2,1)

        ind = np.diag_indices(ADJ.shape[-1])
        ADJ[:,ind[0], ind[1]] = -200 #There is no sel loop

        return ADJ

class GraphTransformerDecoder_FC_R(torch.nn.Module):
    def __init__(self,input,LargestGraphSize, TypeOFnormLayers = "LayerNorm", hiddenLayer=4):
        super(GraphTransformerDecoder_FC_R, self).__init__()
        self.LargestGraphSize = LargestGraphSize

        layers = [input] + [2048]*hiddenLayer


        layers = layers + [int((LargestGraphSize**2 - LargestGraphSize)/2)]
        if TypeOFnormLayers=="LayerNorm":
            self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i+1],elementwise_affine=False) for i in range(len(layers) - 2)])
        else:
            self.normLayers = torch.nn.ModuleList([torch.nn.BatchNorm1d(layers[i+1],affine=None, momentum=None) for i in range(len(layers) - 2)])

        # self.normLayers = torch.nn.ModuleList(
        #     [torch.nn.BatchNorm1d(layers[i + 1], ) for i in range(len(layers) - 2)])
        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1],torch.float32) for i in range(len(layers) - 1)])

    def forward(self, in_tensor, activation= torch.nn.LeakyReLU(0.001)):
        representation = []
        for i in range(len(self.layers)):
            # in_tensor = self.normLayers[i](in_tensor)
            if i==len(self.layers)-1:
                in_tensor = self.layers[i](torch.stack(representation).sum(0))
            else:
                in_tensor = self.layers[i](in_tensor)
                in_tensor = activation(in_tensor)
                in_tensor = self.normLayers[i](in_tensor)
                representation.append(in_tensor)


        ADJ = torch.zeros((in_tensor.shape[0],self.LargestGraphSize,self.LargestGraphSize),device=in_tensor.device)
        ADJ[:,torch.tril_indices(self.LargestGraphSize,self.LargestGraphSize,-1)[0],torch.tril_indices(self.LargestGraphSize,self.LargestGraphSize,-1)[1]] = in_tensor
        ADJ = ADJ + ADJ.permute(0,2,1)

        ind = np.diag_indices(ADJ.shape[-1])
        ADJ[:,ind[0], ind[1]] = -200 #There is no sel loop

        return ADJ

class PositionalDecoder(torch.nn.Module):
    def __init__(self,input,SubGraphNodeNum, TypeOFnormLayers = "LayerNorm", node_vec_dim=1024,):
        super(PositionalDecoder, self).__init__()
        self.SubGraphNodeNum = SubGraphNodeNum
        self.directed = directed
        layers = [input] + [1024,1024,2048 ]

        if TypeOFnormLayers=="LayerNorm":
            self.normLayers = torch.nn.ModuleList([torch.nn.LayerNorm(layers[i+1],elementwise_affine=False) for i in range(len(layers) - 1)])
        else:
            self.normLayers = torch.nn.ModuleList([torch.nn.BatchNorm1d(layers[i+1],affine=None, momentum=None) for i in range(len(layers) - 1)])

        self.layers = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1],torch.float32) for i in range(len(layers) - 1)])

        layers = [layers[-1]+2*node_vec_dim,1]
        self.dense_decoder =  torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i + 1],torch.float32) for i in range(len(layers) - 1)])

        self.edges_embeding = torch.rand((torch.tril_indices(SubGraphNodeNum,SubGraphNodeNum,-1)[0].shape[0], node_vec_dim*2))*2-1
        self.edges_embeding = self.edges_embeding.unsqueeze(0).to(device)

    def forward(self, graph_embeding,  activation= torch.nn.LeakyReLU(0.001)):

        for i in range(len(self.layers)):
            graph_embeding = self.layers[i](graph_embeding)
            graph_embeding = activation(graph_embeding)
            graph_embeding = self.normLayers[i](graph_embeding)

        # repeat the graph embeding
        graph_embeding =graph_embeding.unsqueeze(1)
        graph_embeding =graph_embeding.repeat(1,self.edges_embeding.shape[1],1)
        # concat the positions embedding with graph embedding
        graph_embeding = torch.cat((self.edges_embeding.repeat(graph_embeding.shape[0],1,1),graph_embeding),-1)

        # apply the last layer
        in_tensor = self.dense_decoder  [-1](graph_embeding)
        ADJ = torch.zeros((in_tensor.shape[0],SubGraphNodeNum,SubGraphNodeNum),device=in_tensor.device)
        ADJ[:,torch.tril_indices(SubGraphNodeNum,SubGraphNodeNum,-1)[0],torch.tril_indices(SubGraphNodeNum,SubGraphNodeNum,-1)[1]] = in_tensor.reshape(graph_embeding.shape[0],-1)
        ADJ = ADJ + ADJ.permute(0,2,1)
        return ADJ



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
        self.min_dgree = ker.get("min_dgree")
        self.min_reach = ker.get("min_reach")
        self.largest_graph_size = ker.get("largest_graph_size")
        self.kernel_type = ker.get("kernel_type")
        self.normalized = ker.get("normalized")
        self.num_of_steps = ker.get("step_num")
        self.steps_of_reachability =ker.get("steps_of_reachability")
        self.reach_act = ker.get("reach_act")
        self.ignoreCycles =ker.get("ignore_cycles")
        kernel_set = set(self.kernel_type)

        if "in_degree_dist" in kernel_set or "out_degree_dist" in kernel_set:
            self.degree_hist = ker.get("degree_hist")
            self.max_dgree = ker.get("max_dgree")

        if "HistogramOfRandomWalks" in kernel_set:
            self.num_of_steps = ker.get("step_num")
            self.walks_hist = ker.get("HistogramOfRandomWalks")
            self.max_hists = ker.get("max_hists")
        if "HistogramOfCycles" in kernel_set:
            self.Cycles_hist = ker.get("cyclesHistogram")
            self.max_size_of_cyc = ker.get("max_size_of_cyc")
            self.max_number_of_cycle = ker.get("max_number_of_cycle")
        if "ReachabilityInKsteps" in kernel_set:
            self.ReachabilityHist =ker.get("ReachabilityInKsteps")
            self.steps_of_reachability =ker.get("steps_of_reachability")
            self.MAx_reachability = ker.get("MAx_reachability")

        if "clusteringCoefficient" in kernel_set:
            self.clusteringCoefficientHist = ker.get("clusteringCoefficientHist")

    def forward(self,adj,perturbed_stat=None):
        if perturbed_stat!=None:
            return self.aggregator(perturbed_stat)
        vec = self.kernel_function(adj)
        # return self.hist(vec)
        return vec

    def aggregator(self,perturbed_stat):
        vec = []  # feature vector
        for key,value in perturbed_stat.items():
            if "in_degree_dist" == key:
                re = self.degree_hist(value, self.max_dgree)
                vec.append(re)
            if "HistogramOfCycles" == key:
                re = self.Cycles_hist[0](value)
                vec.append(re)
        return vec

    def kernel_function(self, adj): # TODO: another var for keeping the number of moments
        # ToDo: here we assumed the matrix is symetrix(undirected) which might not
        vec = []  # feature vector
        adj = adj * (1 - torch.eye(adj.shape[-1], adj.shape[-1],device=adj.device))

        # ignore diagonals
        # diag_indx = list(range(adj.shape[-1]))
        # adj[:,diag_indx,diag_indx] = 0


        for kernel in self.kernel_type:
            if "TotalNumberOfTriangles" == kernel:
                vec.append(self.TotalNumberOfTriangles(adj))
            if "in_degree_dist" == kernel:
                re = self.degree_hist(adj.sum(-1),self.max_dgree,self.min_dgree)
                if self.normalized:
                    re = re / adj.shape[-1]
                vec.append(re)
            # if "out_degree_dist" == kernel:
            #     re = self.degree_hist(adj.sum(0))
            #     if self.normalized:
            #         re = re / adj.shape[-1]
            #     vec.append(re)

            if "HistogramOfRandomWalks" == kernel:
                vec.extend(self.RandWalkHist(adj,))
            #
            # if "4-simple-cycle" == kernel:
            #     vec.append(self.NumberOf4Cycles(adj))

            if "HistogramOfCycles" == kernel:
                vec.extend(self.HistogramOfCycleWithLenghtK(adj,2,self.max_size_of_cyc))
            if "ReachabilityInKsteps" == kernel:
                vec.extend(self.ReachabilityInKsteps(adj))

            if "NumberOfVer" == kernel:
                vec.append(self.NumberOfVer(adj))

            # if "twoStarMotifs" == kernel:
            #     vec.append(self.twoStarMotifs(adj))
            #
            # if "clusteringCoefficient" == kernel:
            #     vec.append(self.clusteringCoefficient(adj))

        return vec
    def RandWalkHist(self, adj, ApplyHis=True):
        feature_vec = []
        kth_walks = self.S_randomWalks(adj,self.num_of_steps)
        for h_,walk in enumerate(kth_walks):
            k_th_walk_hist = []
            # for i in range(adj.shape[0]):
            #     # k_th_walk_hist.append(self.walks_hist(torch.triu(walk[i], diagonal=1).view(1, -1)
            #     if ApplyHis:
            #         k_th_walk_hist.append(self.walks_hist[h_](walk[i][torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0],torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[1]].view(1, -1)))
            #     else:
            #         k_th_walk_hist.append((walk[i][torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0],torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[1]].view(1, -1)))

            # k_th_walk_hist.append(self.walks_hist(torch.triu(walk[i], diagonal=1).view(1, -1)
            # since the graph is assumed symetric we consider only the lower triangle
            if ApplyHis:
                feature_vec.append(self.walks_hist[h_](walk[:,torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0],torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[1]],self.max_hists[h_]))
            else:
                feature_vec.append((walk[:,torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0],torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[1]]))

        # if self.normalized:
        #     k_th_walk_hist[-1] = k_th_walk_hist[-1]/ len(torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0])
        # feature_vec.append(torch.cat(k_th_walk_hist))
        # feature_vec.append((k_th_walk_hist))
        # vec.append(walk[:, torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[0],
        # torch.triu_indices(walk.shape[-1], walk.shape[-1], offset=1)[1]].sum(-1))

        # if "cycles_number" == kernel:
        #     vec.extend(self.NumberOFCycleWithLenghtK(adj,3,self.num_of_steps+3))
        return   feature_vec
    def tri_square_count(self, adj):
        two__ = torch.matmul(adj, adj)
        tri_ = torch.matmul(two__, adj)
        squares = torch.matmul(two__, two__)
        return (torch.diagonal(tri_, dim1=1, dim2=2), torch.diagonal(squares, dim1=1, dim2=2))

    def twoStarMotifs(self, adj):
        return torch.matmul(adj, adj).sum((-2,-1)) -adj.sum((-2,-1))

    def clusteringCoefficient(self, adj):
        tri =    (torch.matmul(adj, torch.matmul(adj, adj)))
        tri = torch.diagonal(tri, dim1=1, dim2=2)
        degree = adj.sum(-1)
        all_possibles = degree * (degree - 1)+.000000001

        # all_possibles = torch.clamp(all_possibles, min=.001)
        # mask = all_possibles<=0
        # mask = torch.tensor(mask)+0
        # all_possibles = all_possibles + mask
        clustering_coef = (tri / all_possibles) #* (degree>.001)
        k_th_hist = []
        for i in range(clustering_coef.shape[0]):
            # k_th_walk_hist.append(self.walks_hist(torch.triu(walk[i], diagonal=1).view(1, -1)
            k_th_hist.append(self.clusteringCoefficientHist(clustering_coef[i].view(1, -1)))

        k_th_hist = torch.cat(k_th_hist)
        return k_th_hist

    def S_randomWalks(self, adj, s=4 ):
        """
         this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
        :param s: maximum step; the function will return s-step matrixes from 1 to s in a list
        :param Adj: adjacency matrix of the graph
        :return: a list in whcih the i-th elemnt is the i step transition probablity
        """

        RW_list = []

        p1 = adj
        rw =  torch.matmul(p1, p1 )
        # if ignoreDiag:
        #     mask = 1 - torch.eye(adj.shape[-1]).to(adj.device)
        #     rw = rw * mask

        if self.ignoreCycles :
            diag_indx = list(range(rw.shape[-1]))
            rw[:,diag_indx,diag_indx] = 0
        RW_list.append(rw)
        for i in range(s-1):
            rw = (torch.matmul(p1, RW_list[-1] ))
            # if ignoreDiag:
            #     rw = rw * mask

            if self.ignoreCycles:
                diag_indx = list(range(rw.shape[-1]))
                rw[:,diag_indx,diag_indx] = 0
            RW_list.append(rw)
        return RW_list

    def S_step_trasition_probablity(self, adj, s=4, ):
        """
         this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
        :param s: maximum step; the function will return s-step matrixes from 1 to s in a list
        :param Adj: adjacency matrix of the graph
        :return: a list in whcih the i-th elemnt is the i step transition probablity
        """
        # mask = torch.zeros(adj.shape).to(device)

        p1 = adj
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

        tri = torch.diagonal(torch.matmul(adj, torch.matmul(adj, adj)),dim1=-2, dim2=-1)/6
        return tri

    def NumberOf4Cycles(self, adj, ):
        A_two = torch.matmul(adj, adj)
        A_four = torch.matmul(A_two,A_two)
        A_two = A_two * (1 - torch.eye(adj.shape[-1], adj.shape[-1]))
        A_four = ((torch.diagonal(A_four,dim1=-2, dim2=-1).sum(-1)) - (adj).sum([-2,-1]) - 2* A_two.sum([-2,-1]))/8
        return A_four

    def NumberOfVer(self, adj):
        return adj.sum([-2,-1]).reshape([-1,1])


    def NumberOFCycleWithLenghtK(self,adj, k_l,k_u, Undirected = False):
        """
         this method take an adjacency matrix and count the number of simple cycle  with lenght [k_l,k_l+1,..., k_u]
        """
        result = []

        Adj_k_th = torch.matmul(adj ,adj )


        for _ in range(2,k_l):
            Adj_k_th = torch.matmul(adj ,Adj_k_th )

        for i in range(k_l,k_u):
            Adj_k_th = torch.matmul(adj, Adj_k_th )
            # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
            result.append((torch.diagonal(Adj_k_th,dim1=-2, dim2=-1)/(i*2)).sum(-1))

        return result #torch.cat(result)

    def HistogramOfCycleWithLenghtK(self,adj, k_l,k_u):
        """
         this method take an adjacency matrix and count the number of simple cycle  with lenght [k_l,k_l+1,..., k_u]
        """
        result = []
        Adj_k_th = torch.matmul(adj ,adj )
        for _ in range(2,k_l):
            Adj_k_th = torch.matmul(adj ,Adj_k_th )
        counter=0
        for i in range(k_l,k_u):
            Adj_k_th = torch.matmul(adj, Adj_k_th )
            # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
            # result.append((torch.diagonal(Adj_k_th, dim1=-2, dim2=-1) / (i * 2)).sum(-1))
            re = self.Cycles_hist[counter](torch.diagonal(Adj_k_th,dim1=-2, dim2=-1)/((i+1)*2))
            if self.normalized:
                re = re / (adj.shape[-1])
            result.append(re)#((i+1)*2)
            counter+=1
        return result

    def TreeStepPathes(self, adj,  ):
        """
         this method take an adjacency matrix and count the number of pathes between each two node with lenght 3; this method return a matrix for each graph
        """

        # to save memory Use ineficient loop
        # tri = torch.diagonal(torch.matmul(p1, torch.matmul(p1, p1)),dim1=-2, dim2=-1)/6
        tri = torch.matmul(adj, torch.matmul(adj, adj))
        return tri

    def TotalNumberOfTriangles(self, adj):
        """
         this method take an adjacency matrix and count the number of triangles in it the corresponding graph
        """

        # to save memory Use ineficient loop
        tri = torch.diagonal(torch.matmul(adj, torch.matmul(adj, adj)),dim1=-2, dim2=-1)/6
        tri = tri.sum(-1).reshape(adj.shape[0],-1)
        if self.normalized:
            tri = tri/adj.shape[0]
        return tri

    # def ReachabilityInKsteps(self, adj):
    #     """
    #      this method take an adjacency matrix and count the histogram of number of nodes which are ereachable from a node
    #     """
    #     result = []
    #     Adj_k_th=adj
    #
    #     # result.append(self.ReachabilityHist(adj).sum(-1))
    #     for k_step in range(self.steps_of_reachability):
    #         # Adj_k_th = torch.matmul(adj, Adj_k_th)
    #         # reaches = Adj_k_th
    #         # reaches = torch.clamp(reaches, min=0, max=1).sum(-1)#result.append(self.ReachabilityHist(Adj_k_th.sum(-1)))
    #         Adj_k_th = torch.matmul(adj, Adj_k_th)
    #         Adj_k_th = torch.clamp(Adj_k_th, max=1)
    #         re = self.ReachabilityHist[k_step](Adj_k_th.sum(-1))
    #         if self.normalized:
    #             re = re / adj.shape[-1]
    #         result.append(re)
    #     return result
    #
    def ReachabilityInKsteps(self, adj,applyHist=True):
        """
         this method take an adjacency matrix and count the histogram of number of nodes which are ereachable from a node
        """
        result = []
        Adj_k_th=adj*1
        accumulative_walks = adj*1
        # result.append(self.ReachabilityHist(adj).sum(-1))
        for k_step in range(self.steps_of_reachability):
            # Adj_k_th = torch.matmul(adj, Adj_k_th)
            # reaches = Adj_k_th
            # reaches = torch.clamp(reaches, min=0, max=1).sum(-1)#result.append(self.ReachabilityHist(Adj_k_th.sum(-1)))
            Adj_k_th = torch.matmul(adj, Adj_k_th)
            accumulative_walks += Adj_k_th
            if self.reach_act == "clamp":
                k_thReach = torch.clamp(accumulative_walks, max=1)
            elif self.reach_act == "tanh":
                act = torch.nn.Tanh()
                k_thReach = act(accumulative_walks)
            elif self.reach_act == "identity":
                pass
            elif self.reach_act == "leaky":
                act = torch.nn.LeakyReLU(negative_slope=0.01)
                k_thReach = act(-(accumulative_walks-1.005))
                k_thReach = -k_thReach+1.005
            else:
                warnings.warn("The reach act is not implemented")
            # reached = torch.clamp((Adj_k_th + reached),max=1)
            # reached = torch.clamp((Adj_k_th ), max=1)

            if applyHist:

                re = self.ReachabilityHist[k_step](k_thReach.sum(-1), self.MAx_reachability[k_step],self.min_reach[k_step])
            else:
                re = k_thReach.sum(-1)
            if self.normalized:
                re = re / adj.shape[-1]
            result.append(re)
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

def EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated= True, _f_name=None, OnlyMax_Con_com=True,numOFsamples=None):
    try:
        generated_graphs = graph_generator(model , mini_batch_size, numOFsamples )
        logs = ""
        if OnlyMax_Con_com==False:
            print("all connected_componnents:")
            graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
            if Save_generated:
                np.save(graph_save_path + 'generatedGraphs_adj_' + str(_f_name) + '.npy', graphs_to_writeOnDisk,
                        allow_pickle=True)
            logs = "all connected_componnents: "
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
            graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in generated_graphs]
            np.save(graph_save_path+'Max_Con_comp_generatedGraphs_adj_'+str(_f_name)+'.npy', graphs_to_writeOnDisk, allow_pickle=True)
            graphPloter(generated_graphs, dir =graph_save_path+"_"+str(_f_name), )

            graphs_to_writeOnDisk = [G.toarray() for  G in test_list_adj]
            np.save(graph_save_path+'_Target_Graphs_adj_'+str(_f_name)+'.npy', graphs_to_writeOnDisk, allow_pickle=True)
    except Exception as e:
        logs = str(e)
    return logs

def get_subGraph_features(org_adj,Pertubed_Local_statistics, subgraphs_indexes, kernel_model,batchSize = 100):
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

    batchSize= min(batchSize,len(org_adj))
    if kernel_model!=None:
        target_kelrnel_val = []
        for i in range(0,len(org_adj),batchSize):
            the_batch_stat=None
            if Pertubed_Local_statistics !=None:
                the_batch_stat = {}
                for key,val in Pertubed_Local_statistics.items():
                    the_batch_stat[key] = val[i:i+batchSize]
            target_kelrnel_val.append(kernel_model(subgraphs[i:i+batchSize],the_batch_stat))
        wholeDatasetStats = []
        for i in range(len(target_kelrnel_val[0])):
            feat = torch.cat([batch[i] for batch in target_kelrnel_val])
            wholeDatasetStats.append(feat)


        wholeDatasetStats = [val.to("cpu") for val in wholeDatasetStats]
    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()

    if len(org_adj)!=wholeDatasetStats[0].shape[0]:
        raise Exception("mismatch in the processed data size")
    return  wholeDatasetStats, subgraphs

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

def OptimizerVAE(reconstructed_kernel_val, target_kernel_val, log_std, mean, alpha ,pca_transMtRX,pca_MeanMtRX,explained_variance_):
    device = reconstructed_kernel_val[0].device
    # loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight)

    norm =    mean.shape[0] * mean.shape[1]
    kl = (1/norm)* -0.5 * torch.sum(1+2*log_std - mean.pow(2)-torch.exp(log_std).pow(2))

    # acc = (reconstructed_adj.round() == targert_adj).sum()/float(reconstructed_adj.shape[0]*reconstructed_adj.shape[1]*reconstructed_adj.shape[2])

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
    return kl ,  kernel_diff, each_kernel_loss, kernels_residul

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
# self_for_none = False
# if (decoder_type)in  ("FCdecoder"):#,"FC_InnerDOTdecoder"
self_for_none = False

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
        train_list_adj = train_list_adj[int(len(val_adj)):]
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

# an objet whcih contain graphs and their representtion
list_graphs.processALL(self_for_none=self_for_none)
#we only consider simple graphs
list_graphs.processed_Xs = None
list_graphs.list_Xs = None
list_graphs.x_s = None

connected_componnes= list_graphs.get__Numbers_of_coonented_componnets()
print("The number of graph with multiple conncted componnent: "+str(sum((np.array(connected_componnes)>1))))
#---------------------------------------------------------------------------------------------------------------------
# preprossing on dataset to calculate histograms' bin center
# ignore_cycles = True
adj_list_tmp = list_graphs.get_adj_list()
adj_list_tmp = [torch.tensor(adj_.todense()) for adj_ in adj_list_tmp]
adj_list_tmp = torch.stack(adj_list_tmp).float().to(device)
kernel_ = kernel(kernel_type = [""],step_num = step_num, steps_of_reachability = steps_of_reachability, ignore_cycles=ignore_cycles,reach_act=reach_act)
# adj_list_tmp = adj_list_tmp * (1 - torch.eye(adj_list_tmp.shape[-1], adj_list_tmp.shape[-1])).to(device)

k_steps = kernel_.S_randomWalks(adj_list_tmp, max_needed_step)

SubGraphNodeNum = subgraphSize if subgraphSize!=None else list_graphs.max_num_nodes
# in_feature_dim = list_graphs.feature_size # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes
max_degree = int(list_graphs.get_max_degree())
print("Max degree and graph size are: " + str(max_degree)+ " ," +str(nodeNum))


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
# initialize soft-histogram functions
Pertubed_Local_statistics= None
if LDP==True:
    import LDP_utils
    _Binner = Binner()
    Raw_degree = adj_list_tmp.sum(-1)
    none_isolated = Raw_degree>0
    perturbed_degree = LDP_utils.DD_RandomMEchanism(Raw_degree[none_isolated], epsilon, delta_f = 1) # perturb the node degree for non-isolated nodes
    graph_collected_degree = torch.zeros_like(Raw_degree).to(Raw_degree.device)
    graph_collected_degree[none_isolated] = perturbed_degree
    # degree_center, degree_width = _Binner.combineBinner(binners, Raw_degree, num_of_bin,
    #                                                     UpperBound=SubGraphNodeNum)
    degree_center, degree_width = _Binner.combineBinner(binners, graph_collected_degree, num_of_bin,
                                                        UpperBound=SubGraphNodeNum)
    max_dgree = perturbed_degree.max()
    degree_hist = Histogram((degree_center).to(device), degree_width.to(device), MembershipFunction)
    # --------------------------
    # cycles Histogram
    cyclesHistogram = []
    max_number_of_cycle = []
    #todo: clean it up; only for triangles
    for cycle_ in range(3 - 2):
        Raw_tri = torch.diagonal(k_steps[cycle_ + 1], dim1=1, dim2=2) / ((cycle_ + 3) * 2)
        perturbed_tri = LDP_utils.DD_RandomMEchanism(Raw_tri[none_isolated], epsilon, delta_f=2)
        graph_collected_tri = torch.zeros_like(Raw_tri).to(Raw_tri.device)
        graph_collected_tri[none_isolated] = perturbed_tri
        # center, binwith = _Binner.combineBinner(binners, Raw_tri, num_of_bin, UpperBound=(SubGraphNodeNum ** (cycle_ + 2)) / ((cycle_ + 3) * 2))
        center, binwith = _Binner.combineBinner(binners, graph_collected_tri, num_of_bin, UpperBound=(SubGraphNodeNum ** (cycle_ + 2)) / ((cycle_ + 3) * 2))
        cyclesHistogram.append(Histogram(center.to(device), binwith.to(device), MembershipFunction))
        max_number_of_cycle.append(graph_collected_tri.max())

    Pertubed_Local_statistics = {"in_degree_dist":graph_collected_degree, "HistogramOfCycles":graph_collected_tri}
    kernel_model = kernel(max_number_of_cycle=max_number_of_cycle,max_dgree = max_dgree, cyclesHistogram=cyclesHistogram, kernel_type = kernl_type, step_num = step_num, steps_of_reachability = steps_of_reachability,degree_hist = degree_hist,
                      ReachabilityInKsteps=None, max_size_of_cyc = max_size_of_cyc, HistogramOfRandomWalks = None, largest_graph_size=SubGraphNodeNum,normalized=data_Normalizer,ignore_cycles=ignore_cycles,reach_act=reach_act)

else:
    _Binner = Binner()
    max_dgree = []
    degree_center, degree_width = _Binner.combineBinner(binners,adj_list_tmp.sum(-1), num_of_bin, UpperBound = SubGraphNodeNum)
    degree_hist = Histogram((degree_center).to(device), degree_width.to(device), MembershipFunction)
    max_dgree = adj_list_tmp.sum(-1).max()
    degree_dom = torch.unique(adj_list_tmp.sum(-1))
    min_dgree= min(degree_dom[degree_dom>0])
    #--------------------------
    bin_center = torch.tensor([[i/SubGraphNodeNum] for i in range(0, SubGraphNodeNum+1)]).to(device)
    bin_width = torch.tensor([[.1] for i in bin_center]).to(device)
    binwith =(1/SubGraphNodeNum)

    tri = (torch.matmul(adj_list_tmp, torch.matmul(adj_list_tmp, adj_list_tmp)))
    tri = torch.diagonal(tri, dim1=1, dim2=2)
    degree = adj_list_tmp.sum(-1)
    all_possibles = degree * (degree - 1)+.000000001


    # clustering_coef = (tri / all_possibles)
    # center, width = _Binner.combineBinner(binners, clustering_coef, num_of_bin, UpperBound = 1)
    # clusteringCoefficientHist = Histogram(center, width, MembershipFunction)
    #--------------------------
    HistogramOfRandomWalks = []
    max_hists = []
    randWalks =  kernel_.RandWalkHist(adj_list_tmp,False)
    for i_ in range(step_num):

        bin_center, binwith = _Binner.combineBinner(binners, randWalks[i_], num_of_bin, UpperBound=SubGraphNodeNum**(i_+1) )
        HistogramOfRandomWalks.append(Histogram(bin_center.to(device),binwith.to(device),MembershipFunction))
        max_hists.append(randWalks[i_].max())
    #--------------------------
    # cycles Histogram
    max_number_of_cycle = []
    cyclesHistogram = []
    for cycle_ in range(max_size_of_cyc-2):
        center, binwith = _Binner.combineBinner(binners, torch.diagonal(k_steps[cycle_+1], dim1=1, dim2=2)/((cycle_+3)*2), num_of_bin, UpperBound =(SubGraphNodeNum**(cycle_+2))/((cycle_+3)*2))
        cyclesHistogram.append(Histogram(center.to(device),binwith.to(device),MembershipFunction))
        max_number_of_cycle.append((torch.diagonal(k_steps[cycle_+1], dim1=1, dim2=2)/((cycle_+3)*2)).max())
    #--------------------------


    reach_hist = []
    MAx_reachability = []
    Min_reachability= []
    featureValuse = kernel_.ReachabilityInKsteps(adj_list_tmp,applyHist=False)
    Adj_k_th = adj_list_tmp
    for i_ in range(steps_of_reachability):
        bin_center, binwith = _Binner.combineBinner(binners, featureValuse[i_], num_of_bin, UpperBound=SubGraphNodeNum*(2) )
        reach_hist.append(Histogram( bin_center.to(device), binwith.to(device),MembershipFunction))
        MAx_reachability.append(featureValuse[i_].max())
        reach_dom = torch.unique(featureValuse[i_])
        min_rach = min(reach_dom[reach_dom > 0])
        Min_reachability.append(min_rach)
# --------------------------

    del k_steps, adj_list_tmp, kernel_
    kernel_model = kernel(max_dgree = max_dgree,min_dgree=min_dgree, min_reach=Min_reachability, max_hists=max_hists, max_number_of_cycle=max_number_of_cycle, MAx_reachability= MAx_reachability, cyclesHistogram=cyclesHistogram, kernel_type = kernl_type, step_num = step_num, steps_of_reachability = steps_of_reachability,degree_hist = degree_hist,  ReachabilityInKsteps=reach_hist, max_size_of_cyc = max_size_of_cyc, HistogramOfRandomWalks = HistogramOfRandomWalks, largest_graph_size=SubGraphNodeNum,normalized=data_Normalizer,ignore_cycles=ignore_cycles,reach_act=reach_act)


num_nodes = list_graphs.max_num_nodes
#ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)


start = timeit.default_timer()
# Parameters
step =0
swith = False

min_loss = float('inf')

list_graphs.shuffle()
if(subgraphSize==None):
    # list_graphs.processALL(self_for_none = self_for_none)
    adj_list = list_graphs.get_adj_list()
    graphFeatures , _ = get_subGraph_features(adj_list,Pertubed_Local_statistics, None, kernel_model)
    # PCA_Analiz(graphFeatures[2])
    whiten = False
    pca_MeanMtRX = [None for x in graphFeatures]
    pca_transMtRX = [None for x in graphFeatures]
    pca_VAr = [None for x in graphFeatures]
    if PCATransformer:
        for f, feature in enumerate(graphFeatures):
            print(functions[f + 1])
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
    graphFeatures = [feature.to(device).float() for feature in graphFeatures]
    list_graphs.set_features(graphFeatures)


#----------------------------------------------------------------------------------------------
# model definition

# get inmput dim

invarients_dim = list_graphs.fet_features_dim()
encoder =FC_Encoder(sum(invarients_dim), [256], graphEmDim, dataNormalizer =in_normed_layer, TypeOFnormLayers=normLayers)

if decoder_type == "SBM":
    decoder = GraphTransformerDecoder(graphEmDim, 1024,nodeNum )
elif decoder_type == "FC":
    decoder = GraphTransformerDecoder_FC(graphEmDim, nodeNum, TypeOFnormLayers=normLayers )
elif decoder_type == "sparse":
    decoder = GraphTransformerDecoder_FC_d(graphEmDim,Max_degree=max_degree, SubGraphNodeNum= nodeNum, TypeOFnormLayers=normLayers )
elif decoder_type == "positional":
    decoder = PositionalDecoder(input=graphEmDim,SubGraphNodeNum=nodeNum)
elif decoder_type == "FC_R":
    decoder = GraphTransformerDecoder_FC_R(graphEmDim, nodeNum, TypeOFnormLayers=normLayers, hiddenLayer = NumDecLayers)

model = kernelGVAE(kernel_model,encoder, decoder, AutoEncoder,graphEmDim=graphEmDim) # parameter namimng, it should be dimentionality of distriburion
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)



if scheduler_type=="OneCyle":
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,6000,7000,8000,9000], gamma=0.5)
    num_itr  = epoch_number*max(int(len(list_graphs.list_adjs)/mini_batch_size),1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_itr, pct_start=.20,final_div_factor=5,div_factor=10.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_itr, pct_start=.25,final_div_factor =4,cycle_momentum=True)
elif scheduler_type=="CyclicLR":
    scheduler= torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr*(0.01), max_lr=lr, step_size_up=2000,  cycle_momentum=False)
elif  scheduler_type=="None":
    scheduler=None
elif  scheduler_type=="cosine":
    num_itr  = epoch_number*max(int(len(list_graphs.list_adjs)/mini_batch_size),1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_itr, pct_start=.5,div_factor=1,final_div_factor =100,cycle_momentum=True)
else:
    raise Exception("Sorry, The schedulaer is not defined")
#------------------------------------------------------------------
## ploting the lreaning rate
# import torch
# import matplotlib.pyplot as plt
# lrs = []
# for i in range(epoch_number):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
#     #     print("Factor = ", round(0.65 ** i,3)," , Learning Rate = ",round(optimizer.param_groups[0]["lr"],3))
#     scheduler.step()
#
# plt.plot(range(epoch_number),lrs)
#------------------------------------------------------------------
print(model)
logging.info(model.__str__())
#---------------------------------------------------------------------------------------------------------------------
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
        adj_target,_, _, _, target_kelrnel,lables = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)
        # target_kelrnel = torch.cat(target_kelrnel, dim=1)
        model.train()
        # if subgraphSize == None:
        reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val,reconstructed_adj_logit = model(torch.cat(target_kelrnel, dim=1))
        kl_loss,  kernel_cost,each_kernel_loss,dim_aspect = OptimizerVAE(generated_kernel_val, target_kelrnel  , post_log_std, post_mean, alpha,pca_transMtRX,pca_MeanMtRX,pca_VAr)
        # if lables!=None:
        #     lbl += lables
        #     rec_graphs += reconstructed_adj
        # graphs_in_this_epoch+=subgraphs

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
        pltr.add_values(step, [ loss.cpu().item(), *each_kernel_loss],tmp, redraw= redraw)  # ["Accuracy", "loss", "AUC"])

        step+=1
        optimizer.zero_grad()
        lrs.append(optimizer.param_groups[0]["lr"])
        loss.backward()

        if keepThebest and min_loss>loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()
        if scheduler!=None:
            scheduler.step()


        if (step+1) % visulizer_step == 0 or (epoch == (epoch_number)-1 and iter==0):
            logging.info("****************************************************")
            model.eval()
            pltr.redraw()
            pltr.save_plot(graph_save_path + "MMD_KernelVGAE_log_plot")
            #----------------------------
            # Plot a reconstructed graph
            dir_generated_in_train = "generated_graph_train/"
            if not os.path.isdir(dir_generated_in_train):
                os.makedirs(dir_generated_in_train)
            rnd_indx = random.randint(0,reconstructed_adj.shape[0]-1)
            sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_matrix(sample_graph)
            #remove isolated nodes
            G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            if not nx.is_empty(G):
            # plotter.plotG(G, "generated" + dataset, file_name=graph_save_path+"ReconstructedSample_At_epoch"+str(epoch))
                G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
                # G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
                graphPloter([G],graph_save_path+"ReconstructedRandomSampleAtEpoch_"+str(epoch))
            #---------------------------------------------------------------------------
            print("RECONSTRUCTED graph vs Input (Maximum connected component):")
            logging.info("RECONSTRUCTED graph vs Input (Maximum connected component):")

            reconstructed_adj = reconstructed_adj.cpu().detach().numpy()
            reconstructed_adj[reconstructed_adj >= 0.5] = 1
            reconstructed_adj[reconstructed_adj < 0.5] = 0
            reconstructed_adj = [nx.from_numpy_matrix(reconstructed_adj[i]) for i in range(reconstructed_adj.shape[0])]
            reconstructed_adj = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                 reconstructed_adj if not nx.is_empty(G)]

            target_set = [nx.from_scipy_sparse_matrix(adj_t) for adj_t in adj_target]
            target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                          target_set if not nx.is_empty(G)]



            reconstr_log = mmd_eval(reconstructed_adj, target_set, diam=True)
            logging.info(reconstr_log)
            logging.info("=====================================================")
            # ---------------------------------------------------------------------------

            if task=="graphGeneration":
                #comparing generated graphs and Validation set
                print("-------------------------------------------------------")
                print("comparision with VALIDATION set:")
                logging.info("comparision with VALIDATION set:")
                EvalTwoSet(model, val_adj[:1000], graph_save_path, Save_generated= verbose,_f_name="val_"+str(epoch), numOFsamples = max(200,len(val_adj[:1000])))
                if verbose:

                    #                     EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=verbose, _f_name="test_"+str(epoch))
                    #                     EvalTwoSet(model, train_list_adj, graph_save_path, Save_generated=verbose, _f_name="train_" + str(epoch))

                    EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=verbose, _f_name="test_"+str(epoch),numOFsamples = max(200,len(test_list_adj)))
                    EvalTwoSet(model, train_list_adj, graph_save_path, Save_generated=verbose, _f_name="train_" + str(epoch),numOFsamples = max(200,len(train_list_adj)))

                # print("-------------------------------------------------------")
                # mmd for each category/label
                if len(rec_graphs) > 0:
                    print("there is a bug here for DD")
                    break
                    mmdBylabel(rec_graphs, graphs_in_this_epoch, lbl)
            logging.info("****************************************************")
            # if ((step+1) % visulizer_step*2):
            #     torch.save(model.state_dict(), graph_save_path+"model_"+str(epoch)+"_"+str(batch))
            model.train()

        k_loss_str=""
        for indx,l in enumerate(each_kernel_loss):
            k_loss_str+=functions[indx+1]+":"
            k_loss_str+=str(l)+".   "

        print("Epoch: {:03d} |Batch: {:03d} | loss: {:05f} |  z_kl_loss: {:05f}".format(
            epoch + 1,batch,  loss.item(), kl_loss.item()),k_loss_str)
        logging.info("Epoch: {:03d} |Batch: {:03d} | loss: {:05f}  | z_kl_loss: {:05f} ".format(
            epoch + 1,batch,  loss.item(),  kl_loss.item()) +" "+ str(k_loss_str))
        last_log = " ".join(["Epoch: {:03d} |Batch: {:03d} | loss: {:05f}  | z_kl_loss: {:05f} ".format(
            epoch + 1,batch,  loss.item(),  kl_loss.item()) +" "+ str(k_loss_str)])
        batch+=1
        # scheduler.step()




print(lrs)
model.eval()
torch.save(model.state_dict(), graph_save_path+"model_"+str(epoch)+"_"+str(batch))

stop = timeit.default_timer()
print("trainning time:", str(stop-start))
logging.info("trainning time: "+ str(stop-start))
# save the train loss for comparing the convergence
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

pltr.save_plot(graph_save_path+"MMD_KernelVGAE_log_plot")


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
# print("comparision with VALIDATION set:")
# EvalTwoSet(model, val_adj, graph_save_path+"_Val_", Save_generated=True, _f_name=epoch)
# EvalTwoSet(model, val_adj, graph_save_path, Save_generated= verbose,_f_name="val_"+str(epoch), numOFsamples = max(200,len(val_adj)))
if task=="graphGeneration" :
    print("-------------------------------------------------------")
    print("comparision with VALIDATION set:")
    EvalTwoSet(model, val_adj, graph_save_path+"Val_", Save_generated=True, _f_name=epoch,numOFsamples=len(val_adj))
    if EvalOnTest==True:
        print("comparision with TEST set:")
        EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated= True,_f_name="generated_Test", OnlyMax_Con_com=False,numOFsamples=len(test_list_adj))
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


        pltr.save_plot(write_them_in + "MMD_KernelVGAE_log_plot")
        log = "Reconstructed: \n  maximum_connected_componnent: "+ reconstr_log

        log += "\nVal: " + EvalTwoSet(model, val_adj, write_them_in , Save_generated=True,
                                      _f_name="Val",numOFsamples=len(val_adj))

        train_list_adj
        log += "\n Evaluation on the Train set:" + EvalTwoSet(model, list_adj, write_them_in, Save_generated=True,
                                                             _f_name="Train", numOFsamples=len(test_list_adj))
        if EvalOnTest:
            log +="\n Evaluation on the test set:" + EvalTwoSet(model, test_list_adj, write_them_in , Save_generated=True,
                                                                _f_name="Test",numOFsamples=len(test_list_adj))

        with open(write_them_in+'MMD.log', 'w') as f:
            f.write(last_log+"\n"+log)

        # drop the reconstructed graph in the last epoch
        # reconstructed_adj = [nx.to_numpy_array(G) for G in reconstructed_adj]
        # np.save(write_them_in + 'reconstructed_graph.npy', reconstructed_adj,
        #         allow_pickle=True)
        # recon_target_set = [nx.to_numpy_array(G) for G in target_set]
        # np.save(write_them_in + 'reconstructed_graph_target.npy', recon_target_set,
        #         allow_pickle=True)
        #
        # graphs_to_writeOnDisk = [G.toarray() for G in train_list_adj]
        # np.save(write_them_in +'_Target_Graphs_adj_Train.npy', graphs_to_writeOnDisk, allow_pickle=True)
        # # drop the validation graph in the last epoch
        # val_adj = [np.array(G) for G in val_adj]
        # np.save(write_them_in + 'Val_graph.npy', target_set,
        #         allow_pickle=True)

        # # drop the test graph in the last epoch
        # if EvalOnTest == True:
        #     target_set = [G.toarray() for G in test_list_adj]
        #     np.save(write_them_in + 'Test_graph.npy', target_set,
        #         allow_pickle=True)
        #
        # # drop generated graphs
        # if EvalOnTest == True:
        #     generated_graphs = graph_generator( model, mini_batch_size)
        #     target_set = [nx.to_numpy_array(G) for G in generated_graphs]
        #     np.save(write_them_in + 'Generated_graphs_test.npy', target_set,
        #             allow_pickle=True)

        # if type(lbl)==list and len(lbl)>0:
        #     print("there is a bug here for DD")
        #     # break
        #     # mmdBylabel(rec_graphs, graphs_in_this_epoch, lbl)
