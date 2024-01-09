import torch
import torch.nn as nn
import math

class MutableLinear(nn.Module):
    def __init__(self, in_feature, out_feature, device='cuda'):
        super(MutableLinear, self).__init__()
        self.device = device

        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.module = nn.Linear(in_feature, out_feature, device=device)
    
    def forward(self, x):
        return self.module(x)
    
    def modify_features(self, new_in_feature, new_out_feature):
        new_module = nn.Linear(new_in_feature, new_out_feature, device=self.device)
        new_module.weight.data[:self.out_feature, :self.in_feature] = self.module.weight.data
        new_module.bias.data[:self.out_feature] = self.module.bias.data
        
        self.module = new_module
        self.in_feature = new_in_feature
        self.out_feature = new_out_feature
    
    def increase_features(self, more_in_feature, more_out_feature):
        new_module = nn.Linear(self.in_feature + more_in_feature, self.out_feature + more_out_feature, device=self.device)
        new_module.weight.data[:self.out_feature, :self.in_feature] = self.module.weight.data
        new_module.bias.data[:self.out_feature] = self.module.bias.data
        
        self.module = new_module
        self.in_feature += more_in_feature
        self.out_feature += more_out_feature