from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from dgl.nn.pytorch import GraphConv
import torch_sparse
from torch_sparse import SparseTensor, matmul

def get_sen(sens, idx_sens_train):
    sens_zeros = torch.zeros_like(sens)
    # print(f'sens={sens}')
    sens_1 = sens 
    sens_0 = (1 - sens) 

    # print(f'idx_sens_train={idx_sens_train.shape}')
    # print(f'idx_sens_train={idx_sens_train.shape}')

    # print(f'sens_1={sens_1.shape}')

    sens_1[idx_sens_train] = sens_1[idx_sens_train] / len(sens_1[idx_sens_train])
    sens_0[idx_sens_train] = sens_0[idx_sens_train] / len(sens_0[idx_sens_train])

    # print(f'sens_1={sens_1.shape}')

    sens_zeros[idx_sens_train] = sens_1[idx_sens_train] - sens_0[idx_sens_train]

    sen_mat = torch.unsqueeze(sens_zeros, dim=0)
    # print(f'sen_mat={sen_mat[0, 0:10]}')
    # print(f'sen_mat={sen_mat[0, 10:20]}')

    return sen_mat

# def sen_norm(sen, edge_index):
#     ## edge_index: unnormalized adjacent matrix
#     ## normalize the sensitive matrix
#     edge_index = torch_sparse.fill_diag(edge_index, 1.0) ## add self loop to avoid 0 degree node
#     deg = torch_sparse.sum(edge_index, dim=1)
#     deg_inv_sqrt = deg.pow(-0.5)
#     sen = torch_sparse.mul(sen, deg_inv_sqrt.view(1, -1)) ## col-wise
#     return sen

def check_sen(edge_index, sen):
    nnz = edge_index.nnz()
    deg = torch.eye(edge_index.sizes()[0]).cuda()
    adj = edge_index.to_dense()
    lap = (sen.t() @ sen).to_dense()
    lap2 = deg - adj
    diff = torch.sum(torch.abs(lap2-lap)) / nnz
    assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'


class FMP(GraphConv):
    r"""Fair message passing layer
    """
    _cached_sen = Optional[SparseTensor]

    def __init__(self, 
                 in_feats: int,
                 out_feats: int,
                 K: int, 
                 lambda1: float = None,
                 lambda2: float = None,
                 L2: bool = True,
                 dropout: float = 0.,
                 cached: bool = False, 
                 **kwargs):

        super(FMP, self).__init__(in_feats, out_feats)
        self.K = K
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L2 = L2
        self.dropout = dropout
        self.cached = cached
        
        # assert add_self_loops == True and normalize == True, ''
        # self.add_self_loops = add_self_loops
        # self.normalize = normalize

        self._cached_sen = None  ## sensitive matrix

        self.propa = GraphConv(in_feats, in_feats, weight=False, bias=False, activation=None)

    def reset_parameters(self):
        self._cached_sen = None


    def forward(self, x: Tensor, 
                g, 
                idx_sens_train,
                edge_weight: OptTensor = None, 
                sens=None) -> Tensor:

        if self.K <= 0: return x

        # assert isinstance(edge_index, SparseTensor), "Only support SparseTensor now"
        # assert edge_weight is None, "edge_weight is not supported yet, but it can be extented to weighted case"

        # self.unnormalized_edge_index = edge_index

        cache = self._cached_sen
        if cache is None:
            sen_mat = get_sen(sens=sens, idx_sens_train=idx_sens_train)               ## compute sensitive matrix

            if self.cached:
                self._cached_sen = sen_mat
                self.init_z = torch.zeros((sen_mat.size()[0], x.size()[-1])).cuda()
        else:
            sen_mat = self._cached_sen # N,

        hh = x
        x = self.emp_forward(g, x=x, hh=hh, sen=sen_mat, K=self.K)
        return x


    def emp_forward(self, g, x, hh, K, sen):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1/(1+lambda2)
        beta = 1/(2*gamma)

        # if lambda1 > 0: 
            # z = self.init_z.detach()


        for _ in range(K):

            if lambda2 > 0:
                ## simplied as the following if gamma = 1/(1+lambda2):
                y = gamma * hh + (1-gamma) * self.propa(g, feat=x)
            else:
                y = gamma * hh + (1-gamma) * x # y = x - gamma * (x - hh)

            if lambda1 > 0:
                # print(f'sen={sen.shape}')
                # print(f'y={y.shape}')

                ### initilize node representation z
                z = sen @ F.softmax(y, dim=1) / (gamma * sen @ sen.t())

                
                # print(f'z={z.shape}')
                
                ### forward correction
                x_bar0 = sen.t() @ z
                x_bar1 = F.softmax(x_bar0, dim=1) ## node * feature

                correct = x_bar0 * x_bar1 

                coeff = torch.sum(x_bar0 * x_bar1, 1, keepdim=True)
                correct = correct - coeff * x_bar1

                x_bar = y - gamma * correct
                z_bar  = z + beta * (sen @ F.softmax(x_bar, dim=1))
                # z_bar  = sen @ F.softmax(x_bar, dim=1)
                if self.L2:
                    z  = self.L2_projection(z_bar, lambda_=lambda1, beta=beta)
                else:
                    z  = self.L1_projection(z_bar, lambda_=lambda1)
                
                x_bar0 = sen.t() @ z
                x_bar1 = F.softmax(x_bar0, dim=1) ## node * feature
                
                

                correct = x_bar0 * x_bar1 
                coeff = torch.sum(x_bar0 * x_bar1, 1, keepdim=True)
                # print(f'coeff={torch.diag(coeff).shape}')
                correct = correct - coeff * x_bar1

                x = y - gamma * correct

                # x_pri = torch.log(F.softmax(y, dim=1) - gamma * x_bar0)
                # x = x_pri + torch.mean(y, 1, True) - torch.mean(x_pri, 1, True)
            else:
                x = y # z=0

            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def L1_projection(self, x: Tensor, lambda_):
        # component-wise projection onto the l∞ ball of radius λ1.
        return torch.clamp(x, min=-lambda_, max=lambda_)
    
    def L2_projection(self, x: Tensor, lambda_, beta):
        # projection on the l2 ball of radius λ1.
        coeff = (2*lambda_) / (2*lambda_ + beta)
        return coeff * x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, lambda1={}, lambda2={}, L2={})'.format(
            self.__class__.__name__, self.K, self.lambda1, self.lambda2, self.L2)