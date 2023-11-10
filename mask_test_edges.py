import random

import numpy as np
import pylab as p
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,average_precision_score


def roc_auc_estimator_onGraphList(pos_edges, negative_edges, reconstructed_adj, origianl_agjacency):
    prediction = []
    true_label = []
    for i,_ in enumerate(reconstructed_adj):
        for edge in pos_edges[i]:
            prediction.append(reconstructed_adj[i][edge[0],edge[1]])
            true_label.append(origianl_agjacency[i][edge[0], edge[1]])

        for edge in negative_edges[i]:
            prediction.append(reconstructed_adj[i][edge[0], edge[1]])
            true_label.append(origianl_agjacency[i][edge[0], edge[1]])

    pred = [1 if x>.5 else 0 for x in prediction]
    auc = roc_auc_score(y_score= prediction, y_true= true_label)
    acc = accuracy_score(y_pred= pred, y_true= true_label, normalize= True)
    ap=average_precision_score(y_score= prediction, y_true= true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc , acc,ap, cof_mtx

def roc_auc_estimator(prediction, true_label):

    pred = np.array(prediction)
    pred[pred>.5] = 1
    pred[pred < .5] = 0
    pred = pred.astype(int)
    # pred = [1 if x>.5 else 0 for x in prediction]

    auc = roc_auc_score(y_score= prediction, y_true= true_label)
    acc = accuracy_score(y_pred= pred, y_true= true_label, normalize= True)
    ap=average_precision_score(y_score= prediction, y_true= true_label)
    cof_mtx = confusion_matrix(y_true=true_label, y_pred=pred)
    return auc , acc,ap, cof_mtx

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# def mask_test_edges(adj):
#     # Function to build test set with 10% positive links
#     # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
#     # TODO: Clean up.
#
#     # Remove diagonal elements
#     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#     adj.eliminate_zeros()
#     # Check that diag is zero:
#     # assert np.diag(adj.todense()).sum() == 0
#     assert adj.diagonal().sum() == 0
#
#     adj_triu = sp.triu(adj)
#     adj_tuple = sparse_to_tuple(adj_triu)
#     edges = adj_tuple[0]
#     edges_all = sparse_to_tuple(adj)[0]
#     num_test = max(int(np.floor(edges.shape[0] / 10.)),2)
#     num_val = max(int(np.floor(edges.shape[0] / 20.)),1)
#
#     all_edge_idx = list(range(edges.shape[0]))
#     np.random.shuffle(all_edge_idx)
#     val_edge_idx = all_edge_idx[:num_val]
#     test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
#     test_edges = edges[test_edge_idx]
#     val_edges = edges[val_edge_idx]
#     train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
#     index = list(range(train_edges.shape[0]))
#     np.random.shuffle(index)
#     train_edges_true = train_edges[index[0:]]
#
#     def ismember(a, b, tol=5):
#         rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
#         return np.any(rows_close)
#
#     test_edges_false = []
#     while len(test_edges_false) < len(test_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], edges_all):
#             continue
#         if test_edges_false:
#             if ismember([idx_j, idx_i], np.array(test_edges_false)):
#                 continue
#
#         test_edges_false.append([idx_i, idx_j])
#
#     val_edges_false = []
#     while len(val_edges_false) < len(val_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], edges_all):
#             continue
#         if ismember([idx_i, idx_j], np.array(test_edges_false)):
#             continue
#         if val_edges_false:
#             if ismember([idx_j, idx_i], np.array(val_edges_false)):
#                 continue
#         val_edges_false.append([idx_i, idx_j])
#
#     train_edges_false = []
#     while len(train_edges_false) < len(train_edges_true):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], edges_all):
#             continue
#         if ismember([idx_i, idx_j], np.array(val_edges_false)):
#             continue
#         if ismember([idx_i, idx_j], np.array(test_edges_false)):
#             continue
#         if train_edges_false:
#             if ismember([idx_j, idx_i], np.array(train_edges_false)):
#                 continue
#         train_edges_false.append([idx_i, idx_j])
#     # print(test_edges_false)
#     # print(val_edges_false)
#     # print(test_edges)
#     assert ~ismember(test_edges_false, edges_all)
#     assert ~ismember(val_edges_false, edges_all)
#     assert ~ismember(val_edges, train_edges)
#     assert ~ismember(test_edges, train_edges)
#     assert ~ismember(val_edges, test_edges)
#
#     data = np.ones(train_edges.shape[0])
#
#     # Re-build adj matrix
#     adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
#     adj_train = adj_train + adj_train.T
#
#     ignore_edges_inx = [list(np.array(val_edges_false)[:,0]),list(np.array(val_edges_false)[:,1])]
#     ignore_edges_inx[0].extend(val_edges[:,0])
#     ignore_edges_inx[1].extend(val_edges[:,1])
#     import copy
#
#     val_edge_idx = copy.deepcopy(ignore_edges_inx)
#     ignore_edges_inx[0].extend(test_edges[:, 0])
#     ignore_edges_inx[1].extend(test_edges[:, 1])
#     ignore_edges_inx[0].extend(np.array(test_edges_false)[:, 0])
#     ignore_edges_inx[1].extend(np.array(test_edges_false)[:, 1])
#
#     # NOTE: these edge lists only contain single direction of edge!
#     val_edges_false = np.array(val_edges_false)
#     test_edges_false = np.array(test_edges_false)
#     train_edges_false = np.array(train_edges_false)
#     return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_edges_true, train_edges_false, val_edge_idx

def mask_test_edges(adj):

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)

    u =adj_tuple[0][:,0]
    v = adj_tuple[0][:,1]
    eids = np.arange(len(v))
    eids = np.random.permutation(eids)
    test_size = max(int(len(eids) * 0.2),2)
    val_size = max(int(len(eids) * 0.1), 1)

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size+val_size]], v[eids[test_size:test_size+val_size]]
    # rest should be train
    train_pos_u, train_pos_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]

    # Find all negative edges and split them for training and testing
    adj_ = sp.coo_matrix((np.ones(len(u)), (u, v)),shape=adj.shape)
    adj_ = adj_+adj_.transpose()
    adj_neg = 1 - adj_.todense() - np.eye(adj_.shape[0])
    # neg_u, neg_v = np.where(adj_neg != 0)
    adj_triu = sp.triu(adj_neg)
    adj_tuple = sparse_to_tuple(adj_triu)

    neg_u = adj_tuple[0][:, 0]
    neg_v = adj_tuple[0][:, 1]


    if len(neg_u)< len(v):# if the graph is dense
        neg_eids = list(range(len(neg_u)))
        random.shuffle(neg_eids)
        test_size_ = max(int(len(neg_eids)/20),1)
        val_size_ = max(int(len(neg_eids)/10),1)
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size_]], neg_v[neg_eids[:test_size_]]
        val_neg_u, val_neg_v = neg_u[neg_eids[test_size_:test_size_+val_size_]], neg_v[neg_eids[test_size_:test_size_+val_size_]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size_+val_size_:]], neg_v[neg_eids[test_size_+val_size_:]]
    else:
        neg_eids = np.random.choice(len(neg_u), len(v), replace=False)
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size+val_size]], neg_v[neg_eids[test_size:test_size+val_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size+val_size:]], neg_v[neg_eids[test_size+val_size:]]

    data = np.ones(train_pos_u.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_pos_u,train_pos_v)), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    val_edges = np.array((val_pos_u, val_pos_v))
    test_edges = np.array((test_pos_u, test_pos_v))
    train_edges = np.array((train_pos_u, train_pos_v))
    val_edges_false = np.array(([val_neg_u,val_neg_v]))
    test_edges_false = np.array(([test_neg_u, test_neg_v]))
    train_edges_false = np.array(([train_neg_u, train_neg_v]))
    return adj_train,  val_edges, val_edges_false, test_edges, test_edges_false, train_edges, train_edges_false
