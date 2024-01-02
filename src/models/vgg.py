import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers import MutableLinear, MutableConv2d
from layers import SchedMutableLinear, SchedMutableConv2d

class MutableVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(MutableVGG16, self).__init__()
        self.num_classes = num_classes

        self.conv1_1 = SchedMutableConv2d([3] * 48, range(16, 64, 1), [3] * 48, [1] * 48, [1] * 48)
        self.conv1_2 = SchedMutableConv2d(range(16, 64, 1), range(16, 64, 1), [3] * 48, [1]  * 48, [1] * 48)
        self.conv2_1 = SchedMutableConv2d(range(16, 64, 1), range(32, 128, 2), [3] * 48, [1]  * 48, [1] * 48)
        self.conv2_2 = SchedMutableConv2d(range(32, 128, 2), range(32, 128, 2), [3] * 48, [1]  * 48, [1] * 48)
        self.conv3_1 = SchedMutableConv2d(range(32, 128, 2), range(64, 256, 4), [3] * 48, [1]  * 48, [1] * 48)
        self.conv3_2 = SchedMutableConv2d(range(64, 256, 4), range(64, 256, 4), [3] * 48, [1]  * 48, [1] * 48)
        self.conv3_3 = SchedMutableConv2d(range(64, 256, 4), range(64, 256, 4), [3] * 48, [1]  * 48, [1] * 48)
        self.conv4_1 = SchedMutableConv2d(range(64, 256, 4), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48)
        self.conv4_2 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48)
        self.conv4_3 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48)
        self.conv5_1 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48)
        self.conv5_2 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48)
        self.conv5_3 = SchedMutableConv2d(range(128, 512, 8), range(128, 512, 8), [3] * 48, [1]  * 48, [1] * 48)
        
        self.fc6 = SchedMutableLinear(range(128 * 7 * 7, 512 * 7 * 7, 8 * 7 * 7), range(1024, 4096, 48))
        self.fc7 = SchedMutableLinear(range(1024, 4096, 48), range(1024, 4096, 48))
        self.fc8 = SchedMutableLinear(range(1024, 4096, 48), [num_classes] * 48)

        self.mutable_steps_left = [48] * (16-1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x

