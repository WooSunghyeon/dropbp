import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import random
import numpy as np
from itertools import zip_longest
import math
from dropbp.layer import DropBP
from dropbp.cpp_extention.allocate_p import allocate_p

class DropBPHandler:
    def __init__(self, model):
        self.model = model
        
    def set_initial_drop_rate(self, p):
        for module in self.model.modules():
            if isinstance(module, DropBP):
                module.p = p
                
    def set_diverse_drop_rate(self, p_list, print_log=False):
        total_dbp_layers = sum(1 for module in self.model.modules() if isinstance(module, DropBP))
        if len(p_list) != total_dbp_layers:
            raise ValueError(f"The number of drop rates ({len(p_list)}) does not match the number of DropBP layers ({total_dbp_layers}).")
        for i, module in enumerate(filter(lambda m: isinstance(m, DropBP), self.model.modules())):
            module.p = p_list[i]
            if print_log:
                print(f"Drop rate of layer {i+1} is set to {p_list[i]}")

    def sensitivity_based_drop_bp(self, backprop, target_probability=0.5, min_rate=0, gradnorm=False):
        total_skbp_layers = sum(1 for module in self.model.modules() if isinstance(module, DropBP))
        zero_rates = [min_rate]*total_skbp_layers
        sensitivities=self.compute_sensitivity(backprop, gradnorm=gradnorm)
        flops = self.extract_flops()
        target_flops = int((1-target_probability)*sum(flops))
        
        drop_rates= allocate_p(torch.tensor(zero_rates, dtype=torch.float),
                            torch.tensor(sensitivities, dtype=torch.float),
                            torch.tensor(flops, dtype=torch.float),
                            target_flops,
                            min_rate)
        drop_rates = drop_rates.cpu().numpy()
        print(drop_rates)
        
        self.set_diverse_drop_rate(drop_rates)
        return sensitivities, drop_rates

    def compute_gradient(self, backprop, gradnorm=False):
        setup_seed(526)
        backprop()
        grad = []
        for p in self.model.parameters():
            if p.grad is not None:
                grad.append(p.grad.norm() if gradnorm else p.grad)
            else:
                grad.append(0)
        return grad

    def compute_sensitivity(self, backprop, gradnorm=False):
        sensitivities = []
        total_dropbp_layers = sum(1 for module in self.model.modules() if isinstance(module, DropBP))
        self.set_initial_drop_rate(0)
        original_grad = self.compute_gradient(backprop, gradnorm)
        for i in range(total_dropbp_layers):
            self.set_diverse_drop_rate([1 if j == i else 0 for j in range(total_dropbp_layers)])
            new_grad = self.compute_gradient(backprop, gradnorm)
            sensitivity = sum(((o - n) ** 2).sum() if hasattr((o - n), 'sum') else (o - n) ** 2 for o, n in zip(original_grad, new_grad))
            sensitivities.append(sensitivity)
            print(f'Layer {i+1}, sensitivity {sensitivity}')
        return sensitivities
                                        
    def extract_flops(self, ):
        return [module.flops for module in self.model.modules() if isinstance(module, DropBP)]

    def extract_count(self, ):
        return [module.count for module in self.model.modules() if isinstance(module, DropBP)]

    def detact_non_grad(self,):
        detach_list = []
        for module in self.model.modules():
            if isinstance(module, DropBP):
                detach_list.append(module.is_detach)
        return all(detach_list)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   

