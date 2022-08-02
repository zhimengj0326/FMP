from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.conv import SGConv
from dgl.nn.pytorch.conv import APPNPConv
import torch.nn as nn
from tqdm import tqdm

import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, size = 50, num_classes=2, num_layer= 10):
        super(MLP, self).__init__()

        self.hidden = nn.ModuleList()
        for _ in range(num_layer-2):
            self.hidden.append(nn.Linear(size, size))

        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)

    def forward(self, x):
        out = F.relu(self.first(x))

        for layer in self.hidden:
            out = F.relu(layer(out))
        
        out = self.last(out)
        return out

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers

        for l in range(1, num_layers-1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))

        self.gat_layers.append(torch.nn.Linear(num_hidden * heads[-2], num_classes))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers-1):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](h)
        # logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

    def head_forward(self, h):
        logits = self.gat_layers[-1](h)
        return logits

# class GAT(nn.Module):
#     def __init__(self,
#                  g,
#                  num_layers,
#                  in_dim,
#                  num_hidden,
#                  num_classes,
#                  heads,
#                  activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual):
#         super(GAT, self).__init__()
#         self.g = g
#         self.num_layers = num_layers
#         self.gat_layers = nn.ModuleList()
#         self.activation = activation
#         # input projection (no residual)
#         self.gat_layers.append(GATConv(
#             in_dim, num_hidden, heads[0],
#             feat_drop, attn_drop, negative_slope, False, self.activation))
#         # hidden layers
#         for l in range(1, num_layers):
#             # due to multi-head, the in_dim = num_hidden * num_heads
#             self.gat_layers.append(GATConv(
#                 num_hidden * heads[l-1], num_hidden, heads[l],
#                 feat_drop, attn_drop, negative_slope, residual, self.activation))
#         # output projection
#         self.gat_layers.append(GATConv(
#             num_hidden * heads[-2], num_classes, heads[-1],
#             feat_drop, attn_drop, negative_slope, residual, None))

#         # self.gat_layers.append(torch.nn.Linear(num_hidden * heads[-2], num_classes))

#     def forward(self, inputs):
#         h = inputs
#         for l in range(self.num_layers):
#             # print(f'shape h={h.shape}')
#             h = self.gat_layers[l](self.g, h).flatten(1)
#         # for i, layer in enumerate(self.gat_layers[:-1]):
#         #     h = layer(self.g, h).flatten(1)
#         # output projection
#         logits = self.gat_layers[-1](self.g, h).mean(1)
#         # print(f'shape logits={logits.shape}')
#         # logits = self.gat_layers[-1](self.g, h).mean(1)
#         return logits

#     def head_forward(self, h):
#         logits = self.gat_layers[-1](self.g, h).mean(1)
#         return logits

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        # self.layers.append(GraphConv(n_hidden, n_classes))
        # self.dropout = nn.Dropout(p=dropout)
        self.layers.append(torch.nn.Linear(n_hidden, n_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers[:-1]):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(self.g, h)
        
        # logits = self.layers[-1](self.g, h)
        logits = self.layers[-1](h)
        return logits

    def head_forward(self, h):
        logits = self.layers[-1](self.g, h)
        return logits

class SGC(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 n_hidden,
                 k=2,
                 cached=True):
        super(SGC, self).__init__()

        self.g = g
        # input layer
        self.layers = SGConv(in_feats, n_hidden, k, cached)

        self.head = torch.nn.Linear(n_hidden, n_classes)
    
    def head_forward(self, h):
        logits = self.head(h)
        return logits

    def forward(self, features):
        h = self.layers(self.g, features)
        logits = self.head(h)
        return logits

class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_layers,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))


        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            # print(f'layer={layer}')
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h