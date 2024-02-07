# DropBP: Accelerating Fine-Tuning of Large Language Models
This is the official repository for DropBP: Accelerating Fine-Tuning of Large Language Models.

## Abstract

<div align="center">
    <img width="500" alt="Overview" src="https://github.com/WooSunghyeon/dropbp/assets/85105077/de3c0684-4d41-40a7-a55b-a6cb193fe43d">
</div>

Fine-tuning is essential for enhancing the capabilities of Large Language Models (LLMs), tailoring them to specific tasks or instructions. However, the training process typically involves both forward and backward propagation, demanding substantial computational cost. While layer dropping can reduce this cost by dropping certain layers, it often leads to significant reductions in accuracy because it directly influences the model output, which adversely affects the calculating gradients. To alleviate this issue, we propose Dropping Backward Propagation (DropBP), a novel approach designed to reduce computational cost while maintaining accuracy. DropBP randomly drops layers during the backward propagation, which does not alter output during forward propagation. Moreover, DropBP determines the drop rate for each layer by their sensitivity to stabilize training. DropBP can be applied to all types of fine-tuning based on backpropagation. Specifically, applying DropBP to QLoRA reduces training time by 44%, increases the convergence speed to the identical loss level by 1.5x, and enables training with a 6.2x larger sequence length on a single NVIDIA-A100 80GiB GPU in LLaMA2-70B.

## Install
1. Install [PyTorch](https://pytorch.org/) before installing DropBP library.

2. Build
```bash
pip install -v -e .
```
## Usage

1. Install DropBP layer to your model
+ Note that you have to insert the flops of layers in DropBP layers
+ For instance, in general transformer,
+ FLOPs of attention layers: $4bsh^2+4bhs^2$
+ FLOPs of mlp layers: $16bsh^2$
+ It's okay to input **the ratio of FLOPs** for each layer, rather than exact

```python
import torch
..
from dropbp.layer import DropBP

...
class Block(nn.Modoule): # transformer block
    def __init__(self, ..):
        self.norm_1 = ...
        self.attn = ...
        self.norm_2 = ...
        self.mlp = ...
        
        # Define DropBP layers
        attn_flops = config.hidden_size**2 + config.hidden_size*self.sequence_length 
        mlp_flops = 4*config.hidden_size**2
        self.dropbp_attn = DropBP(layers=[self.norm_1, self.attn], flops=..)
        self.dropbp_mlp = DropBP(layers=[self.norm_2, self.mlp], flops=..)
        ...
    def forward(self, x, ..):
        h = self.attn(self.norm_1(x), ...)
        x = self.dropbp_attn(h)+x   # instead of 'x = h+x'  
        h = self.mlp(self.norm_2(x))
        x = self.dropbp_mlp(h)+x    # instead of 'x = h+x'    
        return x
```
2. Integrate the DropBP API into your training code
```python
import torch
from dropbp.handler import DropBPHandler

model = ... # user define model
optimizer = ... # user define optimizer

dropbp_handler = DropBPHandler(model, optimizer) # define drop handler
dropbp_handler.set_initial_drop_rate(drop_rate) # set a drop rate

# training loop
for iter in ...
    def backprop: # define backprop
        output = model(data)
        loss = loss_func(output, target)
        optimizer.zero_grad() # this line must be present
        loss.backward()

    if iter == int(max_iter * 0.1) # adjust drop rates at 10% of training process 
        dropbp_handler.sensitivity_based_drop_bp(backprop, drop_rate) # it automatically adjusts drop rates
    
    out = model(data)
    loss = loss_func(output,target)
    non_grad = dropbp_handler.detact_non_grad() # detect when all layers are dropped
    if not(non_grad): # exclude the above situation for avoiding error
        loss.backward()
    optimizer.step()
    ...
```

## Applications
Our DropBP library can be very easily integrated with existing training code as:

[Lit-GPT](https://github.com/viqpldem/dropbp/tree/main/lit-gpt)
