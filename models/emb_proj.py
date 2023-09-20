import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential

class MLP(nn.Module):
    def __init__(self, in_feature=28*28, hidden_features=[256], out_feature=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature or self.in_feature
        self.hidden_features = hidden_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.net = nn.ModuleDict()
        net_features = np.hstack((np.array([self.in_feature]), np.array(self.hidden_features), np.array([self.out_feature])))
        assert len(net_features) == len(hidden_features) + 2, "Wrong in MLP class."
        for idx in range(len(net_features) - 1):
            fc_name = "fc" + str(idx+1)
            act_name = "act" + str(idx+1)
            drop_name = "drop" + str(idx+1)
            self.net.update({
                fc_name: nn.Linear(net_features[idx], net_features[idx+1]),
                act_name: self.act,
                drop_name: self.drop
            })
            if idx == len(net_features) - 2: # last layer, no need activation function
                final_act = act_name
                self.net.pop(final_act)
        self.module_list = [module for k, module in self.net.items()]

    def forward(self, x):
        x = checkpoint_sequential(functions=self.module_list, 
                                  segments=1 , 
                                  input=x)
        return x

def EmbeddingProjection():
    model = MLP(in_feature=768, hidden_features=[2048, 2048], out_feature=768, act_layer=nn.ReLU, drop=0.)
    return model