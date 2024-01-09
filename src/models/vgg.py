import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers import MutableLinear, MutableConv2d
from layers import SchedMutableLinear, SchedMutableConv2d

class MutableVGG16(nn.Module):
    def __init__(self, num_classes=10, device='cuda'):
        super(MutableVGG16, self).__init__()
        self.num_classes = num_classes

        self.conv1_1 = SchedMutableConv2d([3] * 48, range(16, 64, 1), [3] * 48, [1] * 48, [1] * 48, device=device)
        self.conv1_2 = SchedMutableConv2d(range(16, 64, 1), range(16, 64, 1), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv2_1 = SchedMutableConv2d(range(16, 64, 1), range(32, 128, 2), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv2_2 = SchedMutableConv2d(range(32, 128, 2), range(32, 128, 2), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv3_1 = SchedMutableConv2d(range(32, 128, 2), range(64, 256, 4), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv3_2 = SchedMutableConv2d(range(64, 256, 4), range(64, 256, 4), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv3_3 = SchedMutableConv2d(range(64, 256, 4), range(64, 256, 4), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv4_1 = SchedMutableConv2d(range(64, 256, 4), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv4_2 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv4_3 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv5_1 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv5_2 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48, device=device)
        self.conv5_3 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48, device=device)
        
        self.fc6 = SchedMutableLinear(range(128 * 1 * 1, 512 * 1 * 1, 8 * 1 * 1), range(1024, 4096, 48), device=device)
        self.fc7 = SchedMutableLinear(range(1024, 4096, 48), range(1024, 4096, 48), device=device)
        self.fc8 = SchedMutableLinear(range(1024, 4096, 48), [num_classes] * 48, device=device)
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        self.mutable_layers = [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.conv4_1, self.conv4_2, self.conv4_3,
            self.conv5_1, self.conv5_2, self.conv5_3,
            self.fc6, self.fc7, self.fc8
        ]
        
        self.mutation_status = [0] * 16
        self.mutation_max_steps = [48] * 16

        self.conv = nn.Sequential(
            self.conv1_1, self.relu,
            self.conv1_2, self.relu,
            self.maxpool,
            self.conv2_1, self.relu,
            self.conv2_2, self.relu,
            self.maxpool,
            self.conv3_1, self.relu,
            self.conv3_2, self.relu,
            self.conv3_3, self.relu,
            self.maxpool,
            self.conv4_1, self.relu,
            self.conv4_2, self.relu,
            self.conv4_3, self.relu,
            self.maxpool,
            self.conv5_1, self.relu,
            self.conv5_2, self.relu,
            self.conv5_3, self.relu,
            self.maxpool,
        ).to(device)

        self.classifier = nn.Sequential(
            self.dropout, self.fc6, self.relu,
            self.dropout, self.fc7, self.relu,
            self.dropout, self.fc8,
        ).to(device)

        self.nn_layers = nn.ModuleList(self.mutable_layers)

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.classifier(x)
        return x

    def set_layer_step(self, layer_idx, step) :
        if layer_idx < 0 or layer_idx >= len(self.mutable_layers) :
            raise ValueError("Invalid layer index")
        if step < 0 or step >= self.mutation_max_steps[layer_idx] :
            raise ValueError("Invalid step")
        
        self.mutable_layers[layer_idx].set_step(step)
        self.mutation_status[layer_idx] = step
    
    def set_step(self, step) :
        if step < 0 or step >= max(self.mutation_max_steps) :
            raise ValueError("Invalid step")
        
        for layer_idx in range(len(self.mutable_layers)) :
            self.set_layer_step(layer_idx, step)
        

