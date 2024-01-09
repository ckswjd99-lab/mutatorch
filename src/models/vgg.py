import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers import MutableLinear, MutableConv2d

class MyMutableVGG(nn.Module):
    def __init__(self):
        super(MyMutableVGG, self).__init__()
        # input: Tensor[batch_size, 3, 32, 32]
        self.conv = nn.ModuleDict({
            'conv1_1': MutableConv2d(3, 16),
            'conv1_2': MutableConv2d(16, 16),
            'conv2_1': MutableConv2d(16, 32),
            'conv2_2': MutableConv2d(32, 32),
            'conv3_1': MutableConv2d(32, 64),
            'conv3_2': MutableConv2d(64, 64),
            'conv3_3': MutableConv2d(64, 64),
        })
        self.fc = nn.ModuleDict({
            'fc1': MutableLinear(64 * 4 * 4, 1024),
            'fc2': MutableLinear(1024, 1024),
            'fc3': MutableLinear(1024, 100),
        })

        self.conv1_1 = self.conv['conv1_1']
        self.conv1_2 = self.conv['conv1_2']
        self.conv2_1 = self.conv['conv2_1']
        self.conv2_2 = self.conv['conv2_2']
        self.conv3_1 = self.conv['conv3_1']
        self.conv3_2 = self.conv['conv3_2']
        self.conv3_3 = self.conv['conv3_3']
        self.fc1 = self.fc['fc1']
        self.fc2 = self.fc['fc2']
        self.fc3 = self.fc['fc3']
        
        self.pool = nn.MaxPool2d(2, 2)

        self.model_config = [3, 16, 16, 32, 32, 64, 64, 64, 1024, 1024, 100]
        self.growth_size = [0, 8, 8, 16, 16, 32, 32, 32, 512, 512, 0]

        self.mutable_layers = [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.fc1, self.fc2, self.fc3
        ]
    
    def forward(self, x):
        # input: Tensor[batch_size, 3, 32, 32]
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)

        # input: Tensor[batch_size, (16-), 16, 16]
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)

        # input: Tensor[batch_size, (32-), 8, 8]
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)

        # input: Tensor[batch_size, (64-), 4, 4]
        x = x.view(-1, self.model_config[7] * (4 * 4))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def grow_all(self, weight_forget=0.0):
        for i in range(len(self.mutable_layers)):
            self.model_config[i] += self.growth_size[i]
        
        self.conv1_1.modify_channels(self.model_config[0], self.model_config[1], weight_forget)
        self.conv1_2.modify_channels(self.model_config[1], self.model_config[2], weight_forget)
        self.conv2_1.modify_channels(self.model_config[2], self.model_config[3], weight_forget)
        self.conv2_2.modify_channels(self.model_config[3], self.model_config[4], weight_forget)
        self.conv3_1.modify_channels(self.model_config[4], self.model_config[5], weight_forget)
        self.conv3_2.modify_channels(self.model_config[5], self.model_config[6], weight_forget)
        self.conv3_3.modify_channels(self.model_config[6], self.model_config[7], weight_forget)
        self.fc1.modify_features(self.model_config[7] * (4 * 4), self.model_config[8])
        self.fc2.modify_features(self.model_config[8], self.model_config[9])
        self.fc3.modify_features(self.model_config[9], self.model_config[10])