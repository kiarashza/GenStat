
import time
start_time = time.monotonic()
import dgl
from input_data import load_data
from mask_test_edges import mask_test_edges, roc_auc_estimator
import plotter
import argparse
import networkx as nx
from GNNs import *
import warnings


torch._set_deterministic(True)





def LinkPrediction(dataset="IMDBBINARY", epoch_number=200,lr = 0.001, encoder="GCN",prediction_layer = "InnerDot",visulizer_step=200, device="cuda:0",filename="",loss_visulizer=False):

    # ************************************************************
    # VGAE frame_work
    class GNN_FrameWork(torch.nn.Module):
        def __init__(self, encoder):
            """
            :param latent_dim: the dimention of each embedded node; |z| or len(z)
            :param numb_of_rel:
            :param decoder:
            :param encoder: The GNN type
            :param mlp_decoder: either apply an multi layer perceptorn on each decoeded embedings
            """
            super(GNN_FrameWork, self).__init__()
            # self.relation_type_param = torch.nn.ParameterList(torch.nn.Parameter(torch.Tensor(2*latent_space_dim)) for x in range(latent_space_dim))
            self.encoder = encoder


        def forward(self, adj, x, ):

            z = self.encoder(adj, x)
            return z

    class Score_Block(torch.nn.Module):
        def __init__(self, predictor):
            """
            :param latent_dim: the dimention of each embedded node; |z| or len(z)
            :param numb_of_rel:
            :param decoder:
            :param encoder: The GNN type
            :param mlp_decoder: either apply an multi layer perceptorn on each decoeded embedings
            """
            super(Score_Block, self).__init__()
            # self.relation_type_param = torch.nn.ParameterList(torch.nn.Parameter(torch.Tensor(2*latent_space_dim)) for x in range(latent_space_dim))
            self.ScorePredictor = predictor


        def forward(self, adj,index_i, index_j ):
            z = self.ScorePredictor(adj, index_i,index_j)
            return z

    # ************************************************************

    # objective Function
    def OptimizerVAE(pos_score, neg_score):

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(scores.device)
        link_prediction_loss = F.binary_cross_entropy_with_logits(scores, labels)



            # z_kl = torch.tensor(0)
        reconstructed_adj = torch.sigmoid(scores)
        acc = ((reconstructed_adj).round() == labels).sum()/float(labels.shape[0])

        return link_prediction_loss, acc, reconstructed_adj,labels


    # ============================================================
    # The main procedure


    original_adj, features = load_data(dataset)

    #-------------------------------------------
    # remove dense and very sparse  graphs
    proces_adj = []
    process_feature = []
    for g_i,graph in enumerate(original_adj):
        if graph.sum() < (graph.shape[0] ** 2 - graph.shape[0]) /3 and (graph.sum() > 6):  # 3 at least 3 negative samples and positive samples

        # if (graph.sum() < graph.shape[0]**2-graph.shape[0]+6) and (graph.sum() > 6): #3 at least 3 negative samples and positive samples
            proces_adj.append(original_adj[g_i])
            process_feature.append(features[g_i])
        else:
            warnings.warn("graph"+"is either too dense or sparse and removed from the loaded dataset")
    features = process_feature
    original_adj = proces_adj
    #-------------------------------------------
    graph_size_list = []
    train_G_list = []
    Batch_size = 0
    Val_Pos = []
    Val_Neg = []
    Test_Pos = []
    Test_Neg = []
    Train_Pos= []
    Train_Neg = []

    def new_index(index, graph_size):
        return  index + graph_size

    for g_i, graph in enumerate(original_adj):

        # shuffling the data, and selecting a subset of it; subgraph_size is used to do the ecperimnet on the samller dataset to insclease development speed
        graph_size_list.append(graph.shape[0])
        indexes = list(range(graph_size_list[-1]))

        #-----------------------------------------
        # adj , feature matrix and  node labels  permutaion
        np.random.shuffle(indexes)
        graph = graph[indexes, :]
        graph = graph[:, indexes]
        original_adj[g_i] = graph # maybe redundant

        features[g_i] = features[g_i][indexes]


        #--------------------------------------------------------------------------------------------- # should be extended for multiple graphs

        # print("processing graph "+ str(g_i))
        # if g_i>507:
        #     print()
        #     pass
        # make train, test and val according to kipf original implementation
        adj_train, val_edges, val_edges_false, test_edges, test_edges_false, train_true, train_false = mask_test_edges(original_adj[g_i])


        Val_Pos.append(new_index(val_edges,Batch_size))
        Val_Neg.append(new_index(val_edges_false,Batch_size))

        Test_Pos.append(new_index(test_edges,Batch_size))
        Test_Neg.append(new_index(test_edges_false,Batch_size))

        Train_Pos.append(new_index(train_true,Batch_size)) # for each graph
        Train_Neg.append(new_index(train_false,Batch_size))
        Batch_size += adj_train.shape[0]

        train_G_list.append(adj_train+ sp.eye(adj_train.shape[0]))# the library does not add self-loops

    #---------------------------------------------------------------------------------------------

    # graph_dgl = dgl.from_scipy(adj_train[0])

    graph_dgl = dgl.batch([dgl.from_scipy(adj_train) for adj_train in train_G_list])
    x_s = torch.tensor(np.concatenate( features)).float()#.to("CUDA:0")
    Pos_edges = np.concatenate(Train_Pos,1).transpose()
    Neg_edges = np.concatenate(Train_Neg,1).transpose()
    Val_Pos_edges = np.concatenate(Val_Pos,1).transpose()
    Val_Neg_edges = np.concatenate(Val_Neg,1).transpose()
    Pos_edges_test = np.concatenate(Test_Pos,1).transpose()
    Neg_edges_test = np.concatenate(Test_Neg,1).transpose()

    # I use this mudule to plot error and loss
    pltr = plotter.Plotter(functions=["loss",  "Accuracy", "Recons Loss",  "AUC"])

    # Check for Encoder and redirect to appropriate function
    if encoder == "GCN":
        encoder_layers = [int(x) for x in [64,128,64]]
        encoder_model = GCN(in_feature=x_s.shape[-1],  layers=encoder_layers)
    elif encoder == "GAT":
        encoder_layers = [int(x) for x in [32, 32]]
        encoder_model = GAT(in_feature=x_s.shape[-1],  layers=encoder_layers)
    elif encoder == "GIN":
        encoder_layers = [int(x) for x in [32, 32,32,32,32]]
        encoder_model = GIN(in_feature=x_s.shape[-1],  layers=encoder_layers)
    elif encoder == "SGC":
        encoder_layers = [int(x) for x in [64, 64]]
        encoder_model = SGC(in_feature=x_s.shape[-1],  layers=encoder_layers)
    else:
        raise Exception("Sorry, this Encoder is not Impemented; check the input args")

    # Check for Decoder and redirect to appropriate function
    # if decoder == ""
    if prediction_layer == "InnerDot":
        decoder_model = InnerProductDecoder()


    # adj_train = torch.tensor(adj_train.todense())  # use sparse man

    # if (type(features) == np.ndarray):
    #     features = torch.tensor(features, dtype=torch.float32)
    # else:
    #     features = torch.tensor(features.todense(), dtype=torch.float32)

    model = GNN_FrameWork( encoder=encoder_model)  # parameter namimng, it should be dimentionality of distriburion
    score_model = Score_Block(decoder_model)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    posible_edges = np.sum([g_size**2 for g_size in graph_size_list])
    pos_wight = torch.true_divide((posible_edges - graph_dgl.num_edges()), graph_dgl.num_edges())  # addrressing imbalance data problem: ratio between positve to negative instance

    # if torch.cuda.is_available():
    #     device = "cuda:"+str(torch.cuda.current_device())
    best_recorded_validation = None
    best_epoch = 0
    # torch.cuda.set_device(0)
    # torch.device('cuda')
    graph_dgl = graph_dgl.to(device)
    x_s = x_s.to(device)
    print(model)
    print(score_model)
    model.to(device)
    score_model.to(device)
    x_s = x_s.to(device)

    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all nodes
        z  = model(graph_dgl, x_s)
        pos_score = score_model(z, Pos_edges[:,0],Pos_edges[:,1])
        neg_score = score_model(z, Neg_edges[:,0],Neg_edges[:,1])
        # compute loss and accuracy
        reconstruction_loss, acc, reconstruct_adj, label= OptimizerVAE(pos_score, neg_score)
        loss = reconstruction_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #--------------------------
        model.eval()

        train_auc, train_acc, train_ap, train_conf = roc_auc_estimator(reconstruct_adj.cpu().detach(), label.cpu())

        VAL_pos_score = score_model(z, Val_Pos_edges[:,0],Val_Pos_edges[:,1])
        VAL_neg_score = score_model(z, Val_Neg_edges[:,0],Val_Neg_edges[:,1])

        val_recons_loss, VAL_acc, val_reconstruct_adj, val_label= OptimizerVAE(VAL_pos_score.detach(), VAL_neg_score.detach())

        val_auc, val_acc, val_ap, val_conf = roc_auc_estimator(val_reconstruct_adj.cpu().detach(), val_label.cpu())


        pltr.add_values(epoch, [loss.item(), train_acc,  reconstruction_loss.item(),  train_auc],
                            [None, val_acc, val_recons_loss.item(), val_auc  # , val_ap
                                ], redraw=False)  # ["Accuracy", "Loss", "AUC", "AP"]




        # Ploting the recinstructed Graph
        if epoch % visulizer_step == 0:
            if loss_visulizer:
                pltr.redraw()
            print("Val conf:", )
            print(val_conf, )
            print("Train Conf:")
            print(train_conf)



        # print some metrics
        print("Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} |  Accuracy: {:03f}".format(
            epoch + 1, loss.item(), reconstruction_loss.item(),  acc), " | AUC:{:5f}".format(train_auc),
            " | AP:{:5f}".format(train_ap))

        print("Val_acc: {:5f} | Val_AUC: {:5f} | Val_AP: {:5f}".format(val_acc, val_auc, val_ap))


    # save the log plot on the current directory

    if loss_visulizer:
        pltr.redraw()
        pltr.save_plot(filename+"_TrainValLoss.tif")
    model.eval()

    # #Loading the best model
    # if dataset not in synthesis_graphs and split_the_data_to_train_test == True:
    #     model = torch.load(PATH)

    print("the best Elbow on validation is " + str(best_recorded_validation) + " at epoch " + str(best_epoch))

    # Link Prediction Task
    print("=====================================")
    print("Result on Link Prediction Task")


    z = model(graph_dgl, x_s)
    pos_score = score_model(z, Pos_edges_test[:, 0], Pos_edges_test[:, 1])
    neg_score = score_model(z, Neg_edges_test[:, 0], Neg_edges_test[:, 1])
    # compute loss and accuracy
    reconstruction_loss, acc, reconstruct_adj, label = OptimizerVAE(pos_score, neg_score)



    Test_auc, Test_acc, Test_ap, conf_mtrx = roc_auc_estimator(reconstruct_adj.cpu().detach(), label.cpu())
    print("Test_acc: {:03f}".format(Test_acc), " | Test_auc: {:03f}".format(Test_auc), " | Test_AP: {:03f}".format(Test_ap))
    print("Confusion matrix: \n", conf_mtrx)

    return  Test_auc,  Test_acc, Test_ap, str(conf_mtrx), train_auc, train_acc, train_ap

    # ------------------------------------------


if __name__ == '__main__':
    LinkPrediction(encoder="GAT")