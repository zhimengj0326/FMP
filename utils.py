#%%
import numpy as np
from numpy.random import beta
import torch.nn.functional as F
import scipy.sparse as sp
import torch
import argparse
import os
import pandas as pd
import dgl
from scipy.special import softmax

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

#%%

#%%
def load_data(path="../dataset/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    print(labels)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_pokec(dataset,sens_attr,predict_attr, path="/data/zhimengj/dataset/pokec/", train_ratio=0.8,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)


    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    # print(f'labels={labels}')
    # print(f'labels={labels.shape}')
    # print(f'label_idx={len(label_idx)}')

    idx_train = label_idx[:int(train_ratio * len(label_idx))]
    idx_val = label_idx[int(train_ratio * len(label_idx)):int( (1+train_ratio)/2 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[int(train_ratio * len(label_idx)):]
        idx_val = idx_test
    else:
        idx_test = label_idx[int( (1+train_ratio)/2 * len(label_idx)):]




    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0]).intersection(set(label_idx))
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)

    # print(f'sens_idx={len(sens_idx)}')
    # print(f'idx_val={idx_val.shape}')
    # print(f'idx_test={idx_test.shape}')
    # print(f'idx_sens_train={idx_sens_train.shape}')

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train

def load_pokec_sub(dataset,sens_attr,predict_attr, path="/data/zhimengj/dataset/pokec/", train_ratio=0.8,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values
    
    #Only nodes for which label and sensitive attributes are available are utilized 
    sens_idx = set(np.where(sens >= 0)[0])
    label_idx = np.where(labels >= 0)[0]
    idx_used = np.asarray(list(sens_idx & set(label_idx)))
    idx_nonused = np.asarray(list(set(np.arange(len(labels))).difference(set(idx_used))))

    features = features[idx_used, :]
    labels = labels[idx_used]
    sens = sens[idx_used]

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    print(f'idx_nonused={idx_nonused}')
    idx_n = idx[idx_nonused]
    idx = idx[idx_used]
    used_ind1 = [i for i, elem in enumerate(edges_unordered[:, 0]) if elem not in idx_n]
    used_ind2 = [i for i, elem in enumerate(edges_unordered[:, 1]) if elem not in idx_n]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_unordered = edges_unordered[intersect_ind, :]

    # build graph
    # idx = np.array(idx_features_labels["user_id"], dtype=int)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_un = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)

    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=int).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)

    adj = sp.coo_matrix((np.ones(edges_un.shape[0]), (edges_un[:, 0], edges_un[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    # print(f'labels={labels}')
    # print(f'labels={labels.shape}')
    # print(f'label_idx={len(label_idx)}')

    idx_train = label_idx[:int(train_ratio * len(label_idx))]
    idx_val = label_idx[int(train_ratio * len(label_idx)):int( (1+train_ratio)/2 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[int(train_ratio * len(label_idx)):]
        idx_val = idx_test
    else:
        idx_test = label_idx[int( (1+train_ratio)/2 * len(label_idx)):]




    # sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0]).intersection(set(label_idx))
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)

    # print(f'sens_idx={len(sens_idx)}')
    # print(f'idx_val={idx_val.shape}')
    # print(f'idx_test={idx_test.shape}')
    # print(f'idx_sens_train={idx_sens_train.shape}')

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):

    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#%%



#%%
def load_pokec_emb(dataset,sens_attr,predict_attr, path="../dataset/pokec/", label_number=1000, seed=19,test_idx=False):
    print('Loading {} dataset from {}'.format(dataset,path))

    graph_embedding = np.genfromtxt(
        os.path.join(path,"{}.embedding".format(dataset)),
        skip_header=1,
        dtype=float
        )
    embedding_df = pd.DataFrame(graph_embedding)
    embedding_df[0] = embedding_df[0].astype(int)
    embedding_df = embedding_df.rename(index=int, columns={0:"user_id"})

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    idx_features_labels = pd.merge(idx_features_labels,embedding_df,how="left",on="user_id")
    idx_features_labels = idx_features_labels.fillna(0)
    #%%

    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    #%%
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:int(0.5 * len(label_idx))]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]




    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

def dp_regularizer(output, labels, sens, idx):
    # val_y = labels[idx]
    val_output = output[idx]

    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]>0

    # print(f'idx_s0={idx_s0.shape}')
    # print(f'idx_s1={idx_s1.shape}')
    # print(f'val_output[idx_s0]={val_output[idx_s0].shape}')

    parity = torch.abs(torch.mean(val_output[idx_s0]) - torch.mean(val_output[idx_s1]))

    # print(f'output={output}')
    # print(f'pred_y={pred_y}')

    return parity

def sample_batch_sen_idx(X, A, y, batch_size, s):    
    # print(f'popu={np.where(A.cpu().numpy()==s)[0]}')
    batch_idx = np.random.choice(np.where(A.cpu().numpy()==s)[0], size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    # batch_x = torch.tensor(batch_x).cuda().float()
    # batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y

def fair_mixup(all_logit, labels, sens, model, alpha = 2):
    idx_s0 = np.where(sens.cpu().numpy()==0)[0]
    idx_s1 = np.where(sens.cpu().numpy()==1)[0]

    # print(f'idx_s0={len(idx_s0)}')
    # print(f'idx_s1={len(idx_s1)}')
    batch_size = min(len(idx_s0), len(idx_s1))

    batch_logit_0, batch_y_0 = sample_batch_sen_idx(all_logit, sens, labels, batch_size, 0)
    batch_logit_1, batch_y_1 = sample_batch_sen_idx(all_logit, sens, labels, batch_size, 1)

    
    gamma = beta(alpha, alpha)

    batch_logit_mix = batch_logit_0 * gamma + batch_logit_1 * (1 - gamma)
    batch_logit_mix = batch_logit_mix.requires_grad_(True)

    output = F.softmax(batch_logit_mix, dim=1)

    # gradient regularization
    gradx = torch.autograd.grad(output.sum(), batch_logit_mix, create_graph=True)[0]

    batch_logit_d = batch_logit_1 - batch_logit_0
    grad_inn = (gradx * batch_logit_d).sum(1)
    E_grad = grad_inn.mean(0)
    loss_reg = torch.abs(E_grad)

    return loss_reg



def fair_metric(output, labels, sens, idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]>0

    # print(f'idx_s0={idx_s0}')

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y>0)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y>0)

    # print(f'idx_s0_y1={idx_s0_y1}')

    pred_y = (output[idx].squeeze()>0.5).type_as(labels).cpu().numpy()

    # print(f'sum(idx_s0)={sum(idx_s0)}')
    # print(f'sum(idx_s1)={sum(idx_s1)}')
    # print(f'sum(idx_s0_y1)={sum(idx_s0_y1)}')
    # print(f'sum(idx_s1_y1)={sum(idx_s1_y1)}')

    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    # print(f'output={output}')
    # print(f'pred_y={pred_y}')

    return parity, equality

class RandNodeSampler(object):
    def __init__(self, src):
        self.src = src
        self.size = len(src)

    def sample(self):
        node_index = np.random.randint(0, self.size, self.size)
        return self.src[node_index]

    def contrast_sample(self, num_samples):

        # node_indexs = []
        size = self.size
        # num_samples = idx_train_fakes.shape[1]

        col_index = torch.randint(num_samples, (size, ))
        row_index = torch.arange(0, size, 1).long()

        # idx_sample = idx_train_fakes[row_index, col_index]

        # src_sentives = sfeature
        # dist = np.abs(src_sentives - np.expand_dims(src_sentives, axis=1))

        # # dist = dist / np.linalg.norm(dist, ord=1, axis=1, keepdims=True)
        # dist = softmax(dist, axis=1)

        # for i in range(size):
        #     node_index = np.random.choice(size, size=1, replace=True, p=dist[i, :])[0]
        #     node_indexs.append(self.src[node_index])

        # print(f'node_indexs={node_indexs}')
        return row_index, col_index

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

