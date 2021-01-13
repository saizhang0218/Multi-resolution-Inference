import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.cpp_extension import load
import numpy as np
import util

#################################################################################
# The cuda kernel for term quantization, three types of term quantization are supported, 
# in the paper we use the optim_code, which is described in the following paper:
# Kung, H. T., Bradley McDanel, and Sai Qian Zhang. 
# "Term quantization: furthering quantization at run time." 
# Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2020.
#################################################################################

tr_cuda = load('tr_cuda', ['kernels/tr_cuda.cpp', 'kernels/tr_cuda_kernel.cu'])
QUANT_TYPES = {'linear': 0, 'hese': 1, 'optim_code': 2}


def weight_quantization(b, power=True):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value_s[idxs].view(shape)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha, num_terms):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            sf = 1 / (2**b - 1)
            input_q = tr_cuda.tr(input_abs, sf, QUANT_TYPES['optim_code'], b, int(16), int(num_terms)).mul(sign)   # term quantization on the scaled weights
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale the weights
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()            
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()            
            return grad_input, grad_alpha, None

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power=True):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        self.power = power if w_bit>2 else False
        self.weight_q = weight_quantization(b=self.w_bit, power=self.power)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight, num_terms):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)      # weights normalization
            weight_q = self.weight_q(weight, self.wgt_alpha, num_terms)
        return weight_q


def act_quantization(b, power=True):
    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha, truncated_terms):
            input=input.div(alpha)  
            input_c = input.clamp(max=1)
            sf = 1 / (2**b - 1)
            input_q = tr_cuda.tr(input_c, sf, QUANT_TYPES['optim_code'], b, 1, truncated_terms)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha, None

    return _uq().apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, bit, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'QuantConv2d'
        self.bit = bit
        self.weight_quant = weight_quantize_fn(w_bit=self.bit, power=True)
        self.act_alq = act_quantization(self.bit, power=True)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))
        
    def forward(self, x, a, k):
        weight_q = self.weight_quant(self.weight, k)
        x = self.act_alq(x, self.act_alpha, a)
        
        return F.conv2d(x, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)
