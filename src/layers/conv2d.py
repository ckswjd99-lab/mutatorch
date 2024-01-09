import torch
import torch.nn as nn
import math

class MutableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, device='cuda'):
        super(MutableConv2d, self).__init__()
        self.device = device

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.module = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, device=device)
    
    def forward(self, x):
        return self.module(x)

    def modify_in_channel(self, new_in_channel, weight_forget=0.0):
        new_module = nn.Conv2d(new_in_channel, self.out_channel, self.kernel_size, self.stride, self.padding, device=self.device)
        new_module.weight.data[:self.out_channel, :self.in_channel, :, :] = self.module.weight.data * (1 - weight_forget)
        new_module.bias.data[:self.out_channel] = self.module.bias.data

        self.module = new_module
        self.in_channel = new_in_channel

    def modify_out_channel(self, new_out_channel, weight_forget=0.0):
        new_module = nn.Conv2d(self.in_channel, new_out_channel, self.kernel_size, self.stride, self.padding, device=self.device)
        new_module.weight.data[:self.out_channel, :self.in_channel, :, :] = self.module.weight.data * (1 - weight_forget)
        new_module.bias.data[:self.out_channel] = self.module.bias.data

        self.module = new_module
        self.out_channel = new_out_channel

    def modify_channels(self, new_in_channel, new_out_channel, weight_forget=0.0):
        new_module = nn.Conv2d(new_in_channel, new_out_channel, self.kernel_size, self.stride, self.padding, device=self.device)
        new_module.weight.data[:self.out_channel, :self.in_channel, :, :] = self.module.weight.data * (1 - weight_forget)
        new_module.bias.data[:self.out_channel] = self.module.bias.data

        self.module = new_module
        self.out_channel = new_out_channel
        self.in_channel = new_in_channel

    def increase_channels(self, more_in_channel, more_out_channel, weight_forget=0.0):
        self.modify_channels(self.in_channel + more_in_channel, self.out_channel + more_out_channel, weight_forget)