import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from functools import partial

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])

    t_max = t.max(dim=-1, keepdim=True)[0]
    t_min = t.min(dim=-1, keepdim=True)[0]
    q_max = 2**n_bits - 1
    scales = torch.max((t_max - t_min) / q_max, torch.tensor(1e-8, device=t.device))
    zero_points = torch.clamp(torch.round(-t_min / scales), 0, 2**n_bits - 1)
    t = scales * torch.clamp(torch.round(t / scales), -zero_points, 2**n_bits -1 -zero_points)
    return t

class INTLinear(nn.Module):
    def __init__(self, 
                weight,
                bias=None,
                per_token_activation_quantization: bool = False,
                kv_output_quant: bool = False,
                ):
        super().__init__()

        self.quantized_weight = nn.Parameter(weight)
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

        self.per_token_activation_quantization = per_token_activation_quantization

        self.kv_output_quant = kv_output_quant
        if self.kv_output_quant:
            self.output_act_quantizer = partial(
                quantize_activation_per_token_absmax, n_bits=8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.per_token_activation_quantization:
            inputs = self.quantize_activation_per_token_absmax(inputs)
        outputs = F.linear(inputs, self.quantized_weight, self.bias)
        if self.kv_output_quant:
            outputs = self.output_act_quantizer(outputs)
        return outputs

    @torch.no_grad()
    def quantize_activation_per_token_absmax(self, t, n_bits=8):
        t_shape = t.shape
        t.view(-1, t_shape[-1])

        t_max = t.max(dim=-1, keepdim=True)[0]
        t_min = t.min(dim=-1, keepdim=True)[0]
        q_max = 2**n_bits - 1
        scales = torch.max((t_max - t_min) / q_max, torch.tensor(1e-8, device=t.device))
        zero_points = torch.clamp(torch.round(-t_min / scales), 0, 2**n_bits - 1)
        t = scales * torch.clamp(torch.round(t / scales), -zero_points, 2**n_bits -1 -zero_points)
        return t

def swapUniformQ(layer, per_token_activation_quantization=False, kv_output_quant=False):
    weight = layer.weight

    if layer.bias is not None:
        bias = layer.bias
    else:
        bias = None

    layer = INTLinear(weight = weight, bias=bias, 
                per_token_activation_quantization = per_token_activation_quantization,
                kv_output_quant=kv_output_quant,
                )

    return layer
