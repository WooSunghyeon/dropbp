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
