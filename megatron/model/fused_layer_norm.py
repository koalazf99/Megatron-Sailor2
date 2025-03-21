# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import importlib
import numbers

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from megatron.core.utils import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

global fused_mix_prec_layer_norm_cuda
fused_mix_prec_layer_norm_cuda = None


class FusedLayerNormAffineFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, weight, bias, normalized_shape, eps):

    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input_ = input.contiguous()
    weight_ = weight.contiguous()
    bias_ = bias.contiguous()
    output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(
        input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
    if False:
        print(input_.shape)
        print(ctx.normalized_shape)
        print(weight_.shape)
        print(bias_.shape)
        print(ctx.eps)

    ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

    return output


  @staticmethod
  def backward(ctx, grad_output):

    input_, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input, grad_weight, grad_bias \
      = fused_mix_prec_layer_norm_cuda.backward_affine(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)

    return grad_input, grad_weight, grad_bias, None, None


class MixedFusedLayerNorm(torch.nn.Module):

  def __init__(self, normalized_shape, eps=1e-5,
               no_persist_layer_norm=True,
               sequence_parallel=False):
        super(MixedFusedLayerNorm, self).__init__()

        global fused_mix_prec_layer_norm_cuda
        # fused_mix_prec_layer_norm_cuda = importlib.import_module(
        #   "fused_mix_prec_layer_norm_cuda")
        fused_mix_prec_layer_norm_cuda = importlib.import_module(
          "fused_layer_norm_cuda")

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [1024, 1536, 2048, 2304, 3072, 3840, 4096,
            5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480,
            24576, 25600, 30720, 32768, 40960, 49152, 65536]
        if normalized_shape not in persist_ln_hidden_sizes or \
                not HAVE_PERSIST_LAYER_NORM:
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel
        
        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

  def reset_parameters(self):

    init.ones_(self.weight)
    init.zeros_(self.bias)

  def forward(self, input):
    if self.no_persist_layer_norm:
        return FusedLayerNormAffineFunction.apply(
          input, self.weight, self.bias, self.normalized_shape, self.eps)
    else:
        output = FastLayerNormFN.apply(
          input, self.weight, self.bias, self.eps)

        # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
        # a populated '_base' field). This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        output = make_viewless_tensor(inp=output,
                                      requires_grad=input.requires_grad,
                                      keep_graph=True)

        return output    


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6,
                 sequence_parallel: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
