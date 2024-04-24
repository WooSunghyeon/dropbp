import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoConfig
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralAttention, MistralFlashAttention2, MistralSdpaAttention, MistralMLP
from transformers.models.mistral.configuration_mistral import MistralConfig
from dropbp.layer import DropBP
from copy import deepcopy

MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "flash_attention_2": MistralFlashAttention2,
    "sdpa": MistralSdpaAttention,
}

class DropBPLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, cutoff_len: int):
        super(LlamaDecoderLayer, self).__init__(config, layer_idx)
        
        qkv_flops = self.hidden_size*self.head_dim+(self.num_heads+2*self.num_key_value_heads)
        attn_flops= self.hidden_size*cutoff_len
        o_flops=self.hidden_size*self.hidden_size
        mlp_flops=3*self.intermediate_size*self.hidden_size
        
        self.dropbp_attn=DropBP([self.input_layernorm, self.self_attn], flops=qkv_flops+attn_flops+o_flops)
        self.dropbp_mlp=DropBP([self.post_attention_layernorm, self.mlp], flops=mlp_flops)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.dropbp_attn(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropbp_mlp(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class DropBPMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int, cutoff_len: int):
    #def __init__(self, config: MistralConfig, cutoff_len: int):
        super(DropBPMistralDecoderLayer, self).__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size//self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.intermediate_size = config.intermediate_size

        qkv_flops=self.hidden_size*self.head_dim+(self.num_heads+2*self.num_key_value_heads)
        attn_flops=self.hidden_size*cutoff_len
        o_flops=self.num_heads*self.head_dim*self.hidden_size
        mlp_flops=3*self.intermediate_size*self.hidden_size
        
        self.dropbp_attn=DropBP([self.input_layernorm, self.self_attn], flops=qkv_flops+attn_flops+o_flops)
        self.dropbp_mlp=DropBP([self.post_attention_layernorm, self.mlp], flops=mlp_flops)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropbp_attn(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropbp_mlp(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

def replace_layer(module, config, layer_idx, cutoff_len):
    if isinstance(module, MistralDecoderLayer):
        target_state_dict   = deepcopy(module.state_dict())
        new_module          = DropBPMistralDecoderLayer(
                                config,
                                layer_idx,
                                cutoff_len
                                )
        new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module

def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

def insert_dropbp(model, cutoff_len=512):
    config=model.config
    layer_idx=0
    for name, module in tuple(model.named_modules()):
        if name:
            recursive_setattr(model, name, replace_layer(module, config, layer_idx, cutoff_len))
            layer_idx +=1
    return model

