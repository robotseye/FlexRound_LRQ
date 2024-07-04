import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class round_ste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if 'tuple' in str(type(pred)):
        pred = pred[0]
    if 'tuple' in str(type(tgt)):
        tgt = tgt[0]
        
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'mse',
                 prob: float = 1.0, flexround: bool = True, 
                 org_weight = None):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric

        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = False
        
        self.flexround = flexround
        if self.flexround:
            self.delta1 = None
            self.delta2 = None 
            self.delta3 = None 
            self.delta4 = None 
            self.delta5 = None 

        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        self.scale_method = scale_method
        self.one_side_dist = None
        self.num = 0 # 100

        self.running_min = None
        self.running_max = None

        self.prob = prob
        self.is_training = False

        if not self.inited:
            self.delta, self.zero_point = self.init_quantization_scale(org_weight.detach(), self.channel_wise)

            if self.flexround:
                if org_weight.size()[0] <= 1024:
                    self.delta1 = torch.nn.Parameter( torch.log(self.delta).detach() )
                    self.delta2 = torch.nn.Parameter(torch.zeros_like(org_weight)) 
                    self.delta3 = torch.nn.Parameter(torch.zeros_like(org_weight[:, 0].unsqueeze(-1)))
                else:
                    self.delta1 = torch.nn.Parameter( torch.log(self.delta).detach() )
                    self.delta2 = torch.nn.Parameter(torch.zeros_like(org_weight[:, :1024])) # 2048
                    self.delta3 = torch.nn.Parameter(torch.zeros_like(org_weight[:1024, :])) # 2048
                    self.delta4 = torch.nn.Parameter(torch.zeros_like(org_weight[:, 0].unsqueeze(-1)))
                    self.delta5 = torch.nn.Parameter(torch.zeros_like(org_weight[0, :].unsqueeze(0)))
                    torch.nn.init.kaiming_uniform_(self.delta3, a=math.sqrt(5))

            self.inited = True
        
    def set_inited(self, inited: bool = True):
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):
        if not self.flexround:
            if not self.sym:
                x_int = round_ste.apply(x / self.delta) 
                x_quant = torch.clamp(x_int, -self.zero_point, self.n_levels - 1 - self.zero_point)
                x_dequant = x_quant * self.delta
            else:
                x_int = round_ste.apply(x / self.delta)
                x_quant = torch.clamp(x_int, -2 ** (self.n_bits - 1), 2 ** (self.n_bits - 1) - 1)
                x_dequant = x_quant * self.delta
        else: 
            if not self.sym:
                x_int = round_ste.apply(x / (self.delta1 + torch.matmul(self.delta2, self.delta3) + self.delta4 + self.delta5).exp()) if self.delta4 is not None else round_ste.apply(x / (self.delta1 + self.delta2 + self.delta3).exp())
                x_quant = torch.clamp(x_int, -self.zero_point, self.n_levels - 1 - self.zero_point)
                x_dequant = x_quant * self.delta1.exp()
            else:
                x_int = round_ste.apply(x / (self.delta1 + torch.matmul(self.delta2, self.delta3) + self.delta4 + self.delta5).exp()) if self.delta4 is not None else round_ste.apply(x / (self.delta1 + self.delta2 + self.delta3).exp())
                x_quant = torch.clamp(x_int, - 2 ** (self.n_bits - 1), 2 ** (self.n_bits - 1) - 1) 
                x_dequant = x_quant * self.delta1.exp()

        return x_dequant

    def lp_loss(self, pred, tgt, p=2.0):
        if 'tuple' in str(type(pred)):
            pred = pred[0]
        if 'tuple' in str(type(tgt)):
            tgt = tgt[0]
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def calculate_qparams(self, min_val, max_val):
        quant_min, quant_max = 0., self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        
        if not self.sym:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        else:
            scale = 2 * torch.max(max_val_pos, torch.abs(min_val_neg)) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = torch.zeros_like(scale)

        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)

        x_int = torch.round(x / delta)

        if not self.sym:
            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta
        else:
            x_quant = torch.clamp(x_int, -2 ** (self.n_bits - 1), 2 ** (self.n_bits - 1) - 1)
            x_float_q = x_quant * self.delta
        
        return x_float_q

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1):  
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1): 
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no':
            best_min, best_max = self.perform_1D_search(x)
        else:
            best_min, best_max = self.perform_2D_search(x)

        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, FlexRound={}, channel_wise={}, symmetric={}, is_training={}, inited={}'.format(
            self.n_bits, self.flexround, self.channel_wise, self.sym, self.is_training, self.inited
        )
