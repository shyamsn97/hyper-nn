import math

import numpy as np
import torch
import torch.nn as nn
from functorch import make_functional, make_functional_with_buffers


def get_weight_chunk_dims(num_target_parameters: int, num_embeddings: int):
    weight_chunk_dim = math.ceil(num_target_parameters / num_embeddings)
    if weight_chunk_dim != 0:
        remainder = num_target_parameters % weight_chunk_dim
        if remainder > 0:
            diff = math.ceil(remainder / weight_chunk_dim)
            num_embeddings += diff
    return weight_chunk_dim


def count_params(module: nn.Module, input_shape=None, inputs=None):
    return sum([np.prod(p.size()) for p in module.parameters()])


class FunctionalParamVectorWrapper(nn.Module):
    """
    This wraps a module so that it takes params in the forward pass
    """

    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.custom_buffers = None
        param_dict = dict(module.named_parameters())
        self.target_weight_shapes = {k: param_dict[k].size() for k in param_dict}

        try:
            _functional, self.named_params = make_functional(module)
        except Exception:
            _functional, self.named_params, buffers = make_functional_with_buffers(
                module
            )
            self.custom_buffers = buffers
        self.functional = [_functional]  # remove params from being counted

    def forward(self, param_vector: torch.Tensor, *args, **kwargs):
        params = []
        start = 0
        for p in self.named_params:
            end = start + np.prod(p.size())
            params.append(param_vector[start:end].view(p.size()))
            start = end
        if self.custom_buffers is not None:
            return self.functional[0](params, self.custom_buffers, *args, **kwargs)
        return self.functional[0](params, *args, **kwargs)
