# Unexpected error: <class 'OSError'>
# Unexpected error: <class 'ZeroDivisionError'>
# degree 0.07535966601684341 clustering 0.048087770412737596 orbits -1

import torch as torch
import torch.nn.functional as F
import numpy as np
from mask_test_edges import mask_test_edges, roc_auc_estimator
from input_data import load_data
import scipy.sparse as sp
import graph_statistics as GS
import plotter
import networkx as nx
import os
import argparse
from util import *
from data import *

import random as random
import time
import timeit

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Kernel VGAE')
parser.add_argument('-e', dest="epoch_number" , default=200, help="Number of Epochs")
parser.add_argument('-v', dest="Vis_step", default=100, help="model learning rate")
parser.add_argument('-redraw', dest="redraw", default=False, help="either update the log plot each step")
parser.add_argument('-lr', dest="lr", default=0.001, help="model learning rate") # for RNN decoder use 0.0001
parser.add_argument('-NSR', dest="negative_sampling_rate", default=1, help="the rate of negative samples which shold be used in each epoch; by default negative sampling wont use")
parser.add_argument('-dataset', dest="dataset", default="ACM", help="possible choices are:  grid, community, citeseer, lobster, DD")#citeceer: ego; DD:protein
parser.add_argument('-NofCom', dest="num_of_comunities", default=32, help="Number of comunites")
parser.add_argument('-s', dest="save_embeddings_to_file", default=False, help="save the latent vector of nodes")
parser.add_argument('-graph_save_path', dest="graph_save_path", default="develope/", help="the direc to save generated synthatic graphs")
parser.add_argument('-f', dest="use_feature" , default=True, help="either use features or identity matrix")
parser.add_argument('-Split', dest="split_the_data_to_train_test" , default=True, help="either use features or identity matrix; for synthasis data default is False")
parser.add_argument('-PATH', dest="PATH" , default="model", help="a string which determine the path in wich model will be saved")
parser.add_argument('-decoder', dest="decoder" , default="FC_InnerDOTdecoder", help="the decoder type,SBMdecoder, FC_InnerDOTdecoder, GRAPHdecoder,FCdecoder,InnerDOTdecoder")
parser.add_argument('-batchSize', dest="batchSize" , default=100, help="the size of each batch")
parser.add_argument('-device', dest="device" , default="cuda:0", help="either use GPU or not if availabel")
parser.add_argument('-model', dest="model" , default="kernel", help="kipf or kernel")
parser.add_argument('-UseGPU', dest="UseGPU" , default=False, help="either use GPU or not if availabel")
parser.add_argument('-task', dest="task" , default="nodeClassification", help="nodeClassification, graphGeneration")
parser.add_argument('-autoencoder', dest="autoencoder" , default=True, help="nodeClassification, graphGeneration")
parser.add_argument('-appendX', dest="appendX" , default=False, help="doese append x to Z for nodeclassification")
args = parser.parse_args()
# torch.autograd.set_detect_anomaly(True)


# **********************************************************************
# setting
print("KernelVGAE SETING: "+str(args))
PATH = args.PATH # the dir to save the with the best performance on validation data
visulizer_step = args.Vis_step
device = args.device
redraw = args.redraw
task = args.task
epoch_number = args.epoch_number
autoencoder = args.autoencoder
lr = args.lr
negative_sampling_rate = args.negative_sampling_rate
hidden_1 = 128  # ?????????????? naming
decoder_type = args.decoder
hidden_2 =  args.num_of_comunities # number of comunities;
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
mini_batch_size = args.batchSize
use_gpu=args.UseGPU

appendX = args.appendX
use_feature = args.use_feature
save_embeddings_to_file = args.save_embeddings_to_file
graph_save_path = args.graph_save_path
split_the_data_to_train_test = args.split_the_data_to_train_test

kernl_type = []

if args.model == "kernel_tri":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "tri", "square"]
    alpha = [1, 1, 1, 1, 1, 1e-06, 1e-06, 0, 0, .001, .001 ]
    step_num = 5
# alpha= [1, 1, 1, 1, 1e-06, 1e-06, 0, 0,.001]
if args.model == "kernel":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist"]
    alpha = [1,1, 1, 1, 1, 1e-06, 1e-06,.001,.001*20]#GRID
    alpha= [10,10, 10, 10, 10, 1e-08*.5, 1e-08*.5,.001,.001] #cora#
    alpha=[24, 24, 24, 24, 24, 5e-09, 5e-09, 0.001, 0.001] #IMDB
    alpha = [10, 10, 10, 10, 10, 5e-09, 5e-09, 0.001, 0.001]#DBLP
    step_num = 5
if args.model == "kipf":
    alpha= [ .001,.001]
    step_num = 0

if autoencoder==True:
    alpha[-1]=0
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

pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log",functions=functions)

synthesis_graphs = {"grid","small_grid", "community", "lobster", "ego"}
if dataset in synthesis_graphs:
    split_the_data_to_train_test = False


# **********************************************************************
class kernelGVAE(torch.nn.Module):
    def __init__(self, in_feature_dim, hidden1,  latent_size, ker, decoder, encoder_fcc_dim = [128] ,autoencoder=False):
        super(kernelGVAE, self).__init__()
        self.first_conv_layer = GraphConvNN(in_feature_dim, hidden1)
        self.second_conv_layer = GraphConvNN(hidden1, hidden1)
        self.stochastic_mean_layer = GraphConvNN(hidden1, latent_size)
        self.stochastic_log_std_layer = GraphConvNN(hidden1, latent_size)
        self.kernel = ker #TODO: bin and width whould be determined if kernel is his

        # self.reset_parameters()
        self.Drop = torch.nn.Dropout(0)
        self.Drop = torch.nn.Dropout(0)
        self.latent_dim = latent_size
        self.mlp = None
        self.decode = decoder
        self.autoencoder = autoencoder

        if None !=encoder_fcc_dim:

            self.fnn =node_mlp(hidden1, encoder_fcc_dim)
            self.stochastic_mean_layer = node_mlp(encoder_fcc_dim[-1], [latent_size])
            self.stochastic_log_std_layer = node_mlp(encoder_fcc_dim[-1], [latent_size])

    def forward(self, graph, features, num_node, ):
        """

        :param graph: normalized adjacency matrix of graph
        :param features: normalized node feature matrix
        :return:
        """
        samples, mean, log_std = self.encode( graph, features,self.autoencoder)
        if type(self.decode)==GRAPHITdecoder:
            reconstructed_adj_logit = self.decode(samples,features)
        elif type(self.decode)==RNNDecoder:
            reconstructed_adj_logit = self.decode(samples,num_node)
        else:
            reconstructed_adj_logit = self.decode(samples)
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logit)
        kernel_value = self.kernel(reconstructed_adj,num_node)

        mask = torch.zeros(graph.shape)

        # removing the effect of none existing nodes
        for i in range(graph.shape[0]):
            reconstructed_adj_logit[i, :, num_node[i]:] = -100
            reconstructed_adj_logit[i, num_node[i]:, :] = -100
            mask[i, :num_node[i], :num_node[i]] = 1
            mean[i,num_node[i]:, :]=  0
            log_std[i,num_node[i]:, :]=  0

        reconstructed_adj = reconstructed_adj * mask.to(device)
        # reconstructed_adj_logit  = reconstructed_adj_logit + mask_logit
        return reconstructed_adj, samples, mean, log_std, kernel_value, reconstructed_adj_logit

    def encode(self, graph, features, autoencoder):
        h = self.first_conv_layer(graph, features)
        h = self.Drop(h)
        h= torch.tanh(h)
        h = self.second_conv_layer(graph, h)
        h = torch.tanh(h)
        if type(self.stochastic_mean_layer) ==GraphConvNN:
            mean = self.stochastic_mean_layer(graph, h)
            log_std = self.stochastic_log_std_layer(graph, h)
        else:
            h = self.fnn(h)
            mean = self.stochastic_mean_layer(h,activation = lambda x:x)
            log_std = self.stochastic_log_std_layer(h,activation = lambda x:x)

        if autoencoder==False:
            sample = self.reparameterize(mean, log_std, node_num)
        else:
            sample = mean*1
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
    def __init__(self, bin_width = None, bin_centers = None):
        super(Histogram, self).__init__()
        self.bin_width = bin_width.to(device)
        self.bin_center = bin_centers.to(device)
        if self.bin_width == None:
            self.prism()
        else:
            self.bin_num = self.bin_width.shape[0]

    def forward(self, vec):
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
        self.decoder = SBMdecoder_(h_size,None)
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

class SBMdecoder_(torch.nn.Module):
    def __init__(self,latent_size, MLP_layers= [128, 256]):
        super(SBMdecoder_, self).__init__()
        self.lamda =torch.nn.Parameter(torch.Tensor(latent_size, latent_size))
        self.mlp = None
        if None !=MLP_layers:
            self.mlp = node_mlp(latent_size, MLP_layers+[latent_size], dropout_rate=.0, normalize=True)
        self.reset_parameters()
    def forward(self, z, activation=lambda x: x):
        # return torch.mm(z1,z2.t())
        if self.mlp != None:
            z = self.mlp(z)
        return activation(torch.matmul(torch.matmul(z, self.lamda), z.permute(0, 2, 1)))
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


class FC_InnerDOTdecoder(torch.nn.Module):
    def __init__(self,input,output,laten_size ,layer=[256,1024,256]):
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
            if ((i!=len(self.layers))):
              h = activation(h)
        h = h.reshape(in_tensor.shape[0], in_tensor.shape[1],-1)
        return torch.matmul(torch.matmul(h,self.lamda), h.permute(0, 2, 1))

    def reset_parameters(self):
        self.lamda = torch.nn.init.xavier_uniform_(self.lamda)
class InnerDOTdecoder(torch.nn.Module):
    def __init__(self):
        super(InnerDOTdecoder,self).__init__()
    # def forward(self,Z):
    #     shape = Z.shape
    #     z = Z.reshape(shape[0],-1)
    #     for i in range(len(self.layers)):
    #         z  = self.layers[i](z)
    #         z = torch.tanh(z)
    #     # Z = torch.sigmoid(Z)
    # return z.reshape(shape[0], shape[-2], shape[-2])
    def forward(self, h):
       return torch.matmul(h, h.permute(0, 2, 1))

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



    def forward(self,adj, num_nodes):
        vec = self.kernel_function(adj, num_nodes)
        # return self.hist(vec)
        return vec

    def kernel_function(self, adj, num_nodes): # TODO: another var for keeping the number of moments
        # ToDo: here we assumed the matrix is symetrix(undirected) which might not
        vec = []  # feature vector
        for kernel in self.kernel_type:
            if "in_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    degree = adj[i,:num_nodes[i],:num_nodes[i]].sum(1).view(1, num_nodes[i])
                    degree_hit.append(self.degree_hist(degree.to(device)))
                vec.append(torch.cat(degree_hit))
            if "out_degree_dist" == kernel:
                degree_hit = []
                for i in range(adj.shape[0]):
                    degree = adj[i, :num_nodes[i], :num_nodes[i]].sum(0).view(1, num_nodes[i])
                    degree_hit.append(self.degree_hist(degree))
                vec.append(torch.cat(degree_hit))
            if "RPF" == kernel:
                raise("should be changed") #ToDo: need to be fixed
                tr_p = self.S_step_trasition_probablity(adj, num_nodes, self.num_of_steps)
                for i in range(len(tr_p)):
                    vec.append(self.hist(torch.diag(tr_p[i])))

            if "trans_matrix" == kernel:
                vec.extend(self.S_step_trasition_probablity(adj, num_nodes, self.num_of_steps))
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

    def S_step_trasition_probablity(self, adj, num_node, s=4, ):
        """
         this method take an adjacency matrix and return its j<s adjacency matrix, sorted, in a list
        :param s: maximum step
        :param Adj: adjacency matrixy of the grap
        :return: a list in whcih the ith elemnt is the i srep transition probablity
        """
        mask = torch.zeros(adj.shape).to(device)
        for i in range(adj.shape[0]):
            mask[i,:num_node[i],:num_node[i]] = 1

        p1 = adj.to(device)
        p1 = p1 * mask
        # ind = torch.eye(adj[0].shape[0])
        # p1 = p1 - ind
        TP_list = []
        p1 = p1*(p1.sum(2).float().clamp(min=1) ** -1).view(adj.shape[0],adj.shape[1], 1)

        # p1[p1!=p1] = 0
        # p1 = p1 * mask

        if s>0:
            # TP_list.append(torch.matmul(p1,p1))
            TP_list.append( p1)
        for i in range(s-1):
            TP_list.append(torch.matmul(p1, TP_list[-1] ))
        return TP_list

def test_(number_of_samples, model ,graph_size,max_size, path_to_save_g, remove_self=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
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
            f_name = path_to_save_g+ str(g_size)+ str(j) + dataset
            # plot and save the generated graph
            plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            plotter.plotG(G, "generated" + dataset, file_name=f_name+"_ConnectedComponnents")
    return generated_graph_list

            # save to pickle file



def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, num_nodes, alpha, reconstructed_adj_logit, pos_wight, norm,node_num, ignore_indexes=None ):
    if ignore_indexes ==None:
        loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight)
    else:
        loss = norm*torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(), targert_adj.float(),pos_weight=pos_wight,
                                                                   reduction='none')
        loss[0][ignore_indexes[1], ignore_indexes[0]] = 0
        loss = loss.mean()
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

def motifs_to_disck(reconstructed_adj, epoch_num, dataset_name):
    reconstructed_adj = reconstructed_adj * (
                1 - torch.eye(reconstructed_adj.shape[1], reconstructed_adj.shape[1]))
    triangels = torch.matmul(torch.matmul(reconstructed_adj, reconstructed_adj.permute(0, 2, 1)),
                             reconstructed_adj.permute(0, 2, 1))
    squres = torch.matmul(triangels, reconstructed_adj.permute(0, 2, 1))
    sgures_num = torch.diagonal(squres, dim1=1, dim2=2)
    triangels_num = torch.diagonal(triangels, dim1=1, dim2=2)

    with open( str(epoch_num) +"_"+dataset_name+"_reconstructedADJ" + '.npy', 'wb') as f:
        np.save(f, reconstructed_adj.cpu().detach().numpy())

    with open(str(epoch_num) +"_"+ dataset_name + '_Circuit_num.npy', 'wb') as f:
        np.save(f, sgures_num.cpu().detach().numpy())

    with open(str(epoch_num) +"_"+ dataset_name + '_triangels_num.npy', 'wb') as f:
        np.save(f, triangels_num.cpu().detach().numpy())

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
ignore_indexes=None
node_label = None
# load the data
if task=="nodeClassification":
    import input_data
    list_adj, list_x, node_label, _, _ = input_data.load_data(dataset)
    list_adj = [list_adj]
    list_x = [list_x]
else:
    list_adj, list_x = list_graph_loader( dataset)

if len(list_adj)==1 and task=="linkPrediction":
    original_adj = list_adj[0].copy()
    list_adj, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_true, train_false, ignore_indexes, val_edge_idx = mask_test_edges(list_adj[0])
    list_adj = [list_adj]



self_for_none = False

if (decoder_type)in  ("FCdecoder"):#,"FC_InnerDOTdecoder"
    self_for_none = True

if len(list_adj)==1:
    test_list_adj=list_adj.copy()
    list_graphs = Datasets(list_adj, self_for_none, list_x)
else:
    list_adj, test_list_adj = data_split(list_adj)
    list_graphs = Datasets(list_adj, self_for_none, None)



degree_center = torch.tensor([[x] for x in range(0, list_graphs.max_num_nodes, 1)])
degree_width = torch.tensor([[.1] for x in range(0, list_graphs.max_num_nodes,1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

kernel_model = kernel(kernel_type = kernl_type, step_num = step_num,
            bin_width= bin_width, bin_center=bin_center, degree_bin_center=degree_center, degree_bin_width=degree_width)
# 225#
in_feature_dim = list_graphs.feature_size # ToDo: consider none Synthasis data

if decoder_type=="SBMdecoder":
    decoder = SBMdecoder_(hidden_2)
elif decoder_type=="FCdecoder":
    decoder= FCdecoder(list_graphs.max_num_nodes*hidden_2,list_graphs.max_num_nodes**2)
elif decoder_type == "InnerDOTdecoder":
    decoder = InnerDOTdecoder()
elif decoder_type == "FC_InnerDOTdecoder":
    decoder = FC_InnerDOTdecoder(list_graphs.max_num_nodes * hidden_2, list_graphs.max_num_nodes *hidden_2, laten_size = hidden_2)
elif decoder_type=="GRAPHITdecoder":
    decoder = GRAPHITdecoder(hidden_2,25)
elif decoder_type=="GRAPHdecoder":
    decoder = GRAPHdecoder(hidden_2)
elif decoder_type=="GRAPHdecoder2":
    decoder = GRAPHdecoder(hidden_2,type="nn",)
elif decoder_type=="RNNDecoder":
    decoder = RNNDecoder(hidden_2)

model = kernelGVAE(in_feature_dim, hidden_1,  hidden_2,  kernel_model,decoder,autoencoder=autoencoder) # parameter namimng, it should be dimentionality of distriburion
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)

num_nodes = list_graphs.max_num_nodes
#ToDo Check the effect of norm and pos weight



start = timeit.default_timer()
# Parameters
step =0

for epoch in range(epoch_number):
    # list_graphs.shuffle()
    batch = 0
    for iter in range(0, len(list_graphs.list_adjs), mini_batch_size):
        from_ = iter
        to_= mini_batch_size*(batch+1) if mini_batch_size*(batch+1)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)
        org_adj,x_s, node_num = list_graphs.get__(from_, to_, self_for_none)
        if(type(decoder))in (FCdecoder, FC_InnerDOTdecoder): #
            node_num = len(node_num)*[list_graphs.max_num_nodes]
        org_adj = torch.cat(org_adj).to(device)
        x_s = torch.cat(x_s)
        pos_wight = torch.true_divide(sum([x**2 for x in node_num])-org_adj.sum(),org_adj.sum())
        model.train()
        target_kelrnel_val = kernel_model(org_adj, node_num)
        reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val,reconstructed_adj_logit = model(org_adj.to(device), x_s.to(device), node_num)
        kl_loss, reconstruction_loss, acc, kernel_cost,each_kernel_loss = OptimizerVAE(reconstructed_adj, generated_kernel_val, org_adj, target_kelrnel_val, post_log_std, post_mean, num_nodes, alpha,reconstructed_adj_logit, pos_wight, 2,node_num, ignore_indexes)


        loss = kernel_cost


        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), None, *each_kernel_loss],tmp, redraw= redraw)  # ["Accuracy", "loss", "AUC"])

        step+=1
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()

        if (epoch+batch) % visulizer_step == 0:
            motifs_to_disck(org_adj, "-1", dataset)
            motifs_to_disck(reconstructed_adj, epoch, dataset)

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
        k_loss_str=""
        for indx,l in enumerate(each_kernel_loss):
            k_loss_str+=functions[indx+3]+":"
            k_loss_str+=str(l)+".   "

        print("Epoch: {:03d} |Batch: {:03d} | loss: {:05f} | reconstruction_loss: {:05f} | z_kl_loss: {:05f} | accu: {:03f}".format(
            epoch + 1,batch,  loss.item(), reconstruction_loss.item(), kl_loss.item(), acc),k_loss_str)

        if(task=="linkPrediction"):
            train_auc, train_acc, train_ap, _ = roc_auc_estimator(train_true, train_false,
                                                                  reconstructed_adj.detach().numpy()[0], original_adj)
            print("Val_acc: {:5f} | Val_AUC: {:5f} | Val_AP: {:5f}".format(train_acc, train_auc, train_ap))
            if split_the_data_to_train_test == True:
                val_auc, val_acc, val_ap, _ = roc_auc_estimator(val_edges, val_edges_false,
                                                                reconstructed_adj.detach().numpy()[0], original_adj)
                print("Val_acc: {:5f} | Val_AUC: {:5f} | Val_AP: {:5f}".format(val_acc, val_auc, val_ap))

        batch+=1
stop = timeit.default_timer()
print("trainning time:", str(stop-start))
# torch.save(model, PATH)


# Node Classification Task
if node_label != None:
    # DBLP Node Label Fix 1.0
    all_senarios_of_input = []

    all_senarios_of_input.append(np.concatenate((prior_samples[0].cpu().detach().numpy(),x_s.detach().numpy()[0]),axis=1))
    all_senarios_of_input.append(prior_samples[0].cpu().detach().numpy())

    if min(node_label) != 0:
        for i in range(len(node_label)):
            node_label[i] -= 1
    for ii, features in enumerate(all_senarios_of_input):
        print("senario"+str(ii))
        print("=====================================")
        print("Result on Node Classification Task")
        import classification
        print("results for NN:")
        labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = classification.NN(
            features, node_label)
        print("Accuracy:{}".format(accuracy),
              "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
              "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
              "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
              "confusion matrix:{}".format(conf_matrix))
        print(report)
        print("******************************************")
        print("results for KNN:")
        labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = classification.knn(
            features, node_label)
        print("Accuracy:{}".format(accuracy),
              "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
              "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
              "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
              "confusion matrix:{}".format(conf_matrix))
        print(report)
        print("******************************************")
        print("results for logistic regression:")
        labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report = classification.logistiic_regression(
            features, node_label)
        print("Accuracy:{}".format(accuracy),
              "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
              "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
              "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
              "confusion matrix:{}".format(conf_matrix))
        print(report)





from stat_rnn import mmd_eval
if dataset in synthesis_graphs:
    generated_graphs = test_(10, model, [x.shape[0] for x in test_list_adj], list_graphs.max_num_nodes, graph_save_path)
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
    reconstructed_adj[reconstructed_adj >= 0.5] = 1
    reconstructed_adj[reconstructed_adj < 0.5] = 0
    reconstructed_adj = nx.from_numpy_matrix(reconstructed_adj[0].cpu().detach().numpy())
    mmd_eval([reconstructed_adj], [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
    generated_graphs = test_(4, model, [x.shape[0] for x in test_list_adj], list_graphs.max_num_nodes, graph_save_path)
    mmd_eval(generated_graphs, [nx.from_numpy_matrix(graph.toarray()) for graph in test_list_adj])
#save the log plot on the current directory
pltr.save_plot(graph_save_path+"KernelVGAE_log_plot")

