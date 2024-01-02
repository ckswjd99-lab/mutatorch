import torch
import torch.nn as nn
import math

class MutableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(MutableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
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
        return nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding)
    
    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias is not None
        )
    
    def modify_in_channels(self, new_in_channels):
        if new_in_channels == self.in_channels:
            return
        elif new_in_channels < self.in_channels:
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(self.out_channels, new_in_channels, *self.kernel_size))
            new_weight.data[:, :, :, :] = self.weight.data[:, :new_in_channels, :, :]

            # Replace old weight matrix
            self.in_channels = new_in_channels
            self.weight = new_weight
        else:    
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(self.out_channels, new_in_channels, *self.kernel_size))
            new_weight.data[:, :self.in_channels, :, :] = self.weight.data[:, :self.in_channels, :, :]

            # Initialize new weights
            nn.init.kaiming_uniform_(self.weight.data[:, self.in_channels:, :, :], a=math.sqrt(5))
            
            # Replace old weight matrix
            self.in_channels = new_in_channels
            self.weight = new_weight

    def modify_out_channels(self, new_out_channels):
        if new_out_channels == self.out_channels:
            return
        elif new_out_channels < self.out_channels:
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(new_out_channels, self.in_channels, *self.kernel_size))
            new_weight.data[:, :, :, :] = self.weight.data[:new_out_channels, :, :, :]

            # Replace old weight matrix
            self.out_channels = new_out_channels
            self.weight = new_weight
        else:    
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(new_out_channels, self.in_channels, *self.kernel_size))
            new_weight.data[:self.out_channels, :, :, :] = self.weight.data[:self.out_channels, :, :, :]

            # Initialize new weights
            nn.init.kaiming_uniform_(self.weight.data[self.out_channels:, :, :, :], a=math.sqrt(5))
            
            # Replace old weight matrix
            self.out_channels = new_out_channels
            self.weight = new_weight

    def modify_kernel_size(self, new_kernel_size):
        if new_kernel_size == self.kernel_size:
            return
        elif new_kernel_size < self.kernel_size:
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *new_kernel_size))
            new_weight.data[:, :, :, :] = self.weight.data[:, :, :new_kernel_size[0], :new_kernel_size[1]]

            # Replace old weight matrix
            self.kernel_size = new_kernel_size
            self.weight = new_weight
        else:    
            # Create new weight matrix
            new_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *new_kernel_size))
            new_weight.data[:, :, :self.kernel_size[0], :self.kernel_size[1]] = self.weight.data[:, :, :, :]

            # Initialize new weights
            nn.init.kaiming_uniform_(self.weight.data[:, :, self.kernel_size[0]:, self.kernel_size[1]:], a=math.sqrt(5))
            
            # Replace old weight matrix
            self.kernel_size = new_kernel_size
            self.weight = new_weight

    def modify_stride(self, new_stride):
        self.stride = new_stride if isinstance(new_stride, tuple) else (new_stride, new_stride)

    def modify_padding(self, new_padding):
        self.padding = new_padding if isinstance(new_padding, tuple) else (new_padding, new_padding)

    def modify_features(self, new_in_channels, new_out_channels, new_kernel_size, new_stride, new_padding):
        self.modify_in_channels(new_in_channels)
        self.modify_out_channels(new_out_channels)
        self.modify_kernel_size(new_kernel_size)
        self.modify_stride(new_stride)
        self.modify_padding(new_padding)
    

class SchedMutableConv2d(MutableConv2d):
    def __init__(self, in_channels_plan, out_channels_plan, kernel_size_plan, stride_plan, padding_plan, bias=True):
        super().__init__(in_channels_plan[0], out_channels_plan[0], kernel_size_plan[0], stride_plan[0], padding_plan[0], bias)
        self.in_channels_plan = in_channels_plan
        self.out_channels_plan = out_channels_plan
        self.kernel_size_plan = kernel_size_plan
        self.stride_plan = stride_plan
        self.padding_plan = padding_plan

        self.total_steps = len(in_channels_plan)
        self.step = 0

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding)
    
    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias is not None
        )
    
    def set_step(self, step):
        self.step = step
        self.modify_features(self.in_channels_plan[step], self.out_channels_plan[step], self.kernel_size_plan[step], self.stride_plan[step], self.padding_plan[step])

    def increment_step(self):
        self.set_step(self.step + 1)
        