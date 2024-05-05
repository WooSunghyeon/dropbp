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
    def __init__(self, flops, p=0):
        super(DropBP, self).__init__()
        self.p=p
        self.flops = flops
        self.count = 0
        self.is_detach = True         
        
    def forward(self, input):
        if not(self.training) or random.random() <= 1-self.p:
            self.is_detach = False
            return input
        else:
            self.count += 1   
            self.is_detach = True
            return input.detach()
