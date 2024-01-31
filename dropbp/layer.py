import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import random
import numpy as np
from itertools import zip_longest
import math
from torch.cuda.amp import custom_fwd, custom_bwd

class DropBP(nn.Module):
    def __init__(self, layers, flops, p=0):
        super(DropBP, self).__init__()
        self.layers=layers
        self.p=p
        self.flops = flops
        self.count = 0
        
    def forward(self, input):
        if not(self.training) or random.random() <= 1-self.p:
            self.is_detach = False
            return input
        else:
            self.count += 1   
            self.is_detach = True
            return input.detach()
        
        
    def adjust_learning_rate(self, optimizer):
        layer_params_ids = {id(p) for layer in self.layers for p in layer.parameters()}
        for param_group in optimizer.param_groups:
            if any(id(p) in layer_params_ids for p in param_group['params']):
                param_group['lr'] *= (1 / (1-self.p))

def truncated_svd(weight, dtype=torch.bfloat16, compression_rate=0):
    if compression_rate == 0:
        return None, None
    
    U, S, V = torch.linalg.svd(weight.double(), full_matrices=False)
    m, n = weight.shape
    num_singular_values = int(m*n*(1-compression_rate)/(m+n))
    
    return U[:, :num_singular_values].to(dtype), (torch.diag(S)[:num_singular_values,:num_singular_values] @ V[:num_singular_values, :]).to(dtype)


class dropfp(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    @custom_fwd
    # bias is an optional argument
    def forward(ctx, input1, input2):
        return input1

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output, grad_output
    
class DropFP(nn.Module):
    def __init__(self,p=0.5):
        super(DropFP, self).__init__()
        self.p=p

    def forward(self, input1, input2):
        if not(self.training) or random.random() <= 1-self.p:
            return input1+input2
        else:
            return dropfp.apply(input1, input2)
