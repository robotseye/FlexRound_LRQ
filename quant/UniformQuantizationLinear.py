import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from quant.UniformAffineQuantizer import * 

class INTLinear(nn.Module):
    def __init__(self, 
                org_weight,
                n_bits: int = 8, symmetric: bool = False, channel_wise: bool = True,
                flexround:bool = False,
                bias=None):
        super().__init__()

        self.org_weight = org_weight
        self.quantized_weight = None
        
        self.weight_quantizer = UniformAffineQuantizer(
                n_bits = n_bits, symmetric = symmetric, channel_wise = channel_wise,
                flexround = flexround, org_weight = org_weight)      

        if not flexround:
            self.weight_quantizer = AdaRoundQuantizer(
                    uaq = self.weight_quantizer, round_mode='learned_hard_sigmoid',
                    weight_tensor= org_weight.data, symmetric = symmetric)
            self.weight_quantizer.soft_targets = True

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.quantized_weight is None:
            weight = self.weight_quantizer(self.org_weight)
        else:
            weight = self.quantized_weight
        return F.linear(inputs, weight, self.bias)

def swapUniformQ(layer, n_bits, num_alpha, channel_wise=True, flexround=False, symmetric=False,
             bool_transpose=False):
    weight = layer.weight

    if layer.bias is not None:
        bias = layer.bias
    else:
        bias = None

    layer = INTLinear(org_weight = weight, n_bits= n_bits, 
                symmetric = symmetric, channel_wise = channel_wise,
                flexround = flexround,
                bias=bias)

    return layer
