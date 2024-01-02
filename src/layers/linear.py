import torch
import torch.nn as nn
import math

class MutableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MutableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def modify_in_features(self, new_in_features):
        if new_in_features == self.in_features:
            return
        elif new_in_features < self.in_features:
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(self.out_features, new_in_features))
            new_weight.data[:, :] = self.weight.data[:, :new_in_features]

            # Replace old weight matrix
            self.in_features = new_in_features
            self.weight = new_weight
        else:    
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(self.out_features, new_in_features))
            new_weight.data[:, :self.in_features] = self.weight.data[:, :self.in_features]

            # Initialize new weights
            nn.init.kaiming_uniform_(self.weight.data[:, self.in_features:], a=math.sqrt(5))
            
            # Replace old weight matrix
            self.in_features = new_in_features
            self.weight = new_weight

    def modify_out_features(self, new_out_features):
        if new_out_features == self.out_features:
            return
        elif new_out_features < self.out_features:
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(new_out_features, self.in_features))
            new_weight.data[:, :] = self.weight.data[:new_out_features, :]

            # Create new bias vector
            if self.bias is not None:
                new_bias = nn.Parameter(torch.Tensor(new_out_features))
                new_bias.data[:] = self.bias.data[:new_out_features]
                self.bias = new_bias

            # Replace old weight matrix
            self.out_features = new_out_features
            self.weight = new_weight
        else:
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(new_out_features, self.in_features))
            new_weight.data[:self.out_features, :] = self.weight.data[:self.out_features, :]

            # Create new bias vector
            if self.bias is not None:
                new_bias = nn.Parameter(torch.Tensor(new_out_features))
                new_bias.data[:self.out_features] = self.bias.data[:self.out_features]
                self.bias = new_bias

            # Initialize new weights
            nn.init.kaiming_uniform_(self.weight.data[self.out_features:, :], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias.data[self.out_features:], -bound, bound)
            
            # Replace old weight matrix
            self.out_features = new_out_features
            self.weight = new_weight
            if self.bias is not None:
                self.bias = new_bias

    def modify_features(self, new_in_features, new_out_features):
        self.modify_in_features(new_in_features)
        self.modify_out_features(new_out_features)

class SchedMutableLinear(MutableLinear):
    def __init__(self, in_features_plan, out_features_plan, bias=True):
        super().__init__(in_features_plan[0], out_features_plan[0], bias)
        
        self.in_features_plan = in_features_plan
        self.out_features_plan = out_features_plan

        self.total_steps = len(in_features_plan)
        self.step = 0

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features_plan[self.step], self.out_features_plan[self.step], self.bias is not None
        )
    
    def set_step(self, step):
        self.step = step
        self.modify_features(self.in_features_plan[step], self.out_features_plan[step])
    
    def increment_step(self):
        self.set_step(self.step + 1)
        