import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from fmp import FMP

class FairGNN(torch.nn.Module):
    def __init__(self, input_size, size, num_classes, num_layer, prop, **kwargs):
        super(FairGNN, self).__init__()

        self.hidden = nn.ModuleList()
        for _ in range(num_layer-2):
            self.hidden.append(nn.Linear(size, size))

        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)

        
        self.prop = prop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, features, g, sens, idx_sens_train):
        x = features
        
        out = F.relu(self.first(x))

        for layer in self.hidden:
            out = F.relu(layer(out))
        
        x = self.last(out)

        x = self.prop(x, sens=sens, g=g, idx_sens_train=idx_sens_train)
        # return F.log_softmax(x, dim=1)
        return x



def get_model(args, data):

    Model = FairGNN

    prop =  FMP(in_feats=data.num_features,
                out_feats=data.num_features,
                K=args.num_layers, 
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                L2=args.L2,
                cached=True)

    model = Model(input_size=data.num_features, 
                    size=args.num_hidden, 
                    num_classes=data.num_classes, 
                    num_layer=args.num_gnn_layer, 
                    prop=prop).cuda()

    return model