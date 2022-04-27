import math

import numpy as np
import torch
import torch.nn as nn
from functorch import make_functional


def get_hidden_weight_generator_dims(num_target_parameters: int, num_embeddings: int):
    hidden_dim = math.ceil(num_target_parameters / num_embeddings)
    if hidden_dim != 0:
        remainder = num_target_parameters % hidden_dim
        if remainder > 0:
            diff = math.ceil(remainder / hidden_dim)
            num_embeddings += diff
    return hidden_dim


def count_params(module: nn.Module, input_shape=None, inputs=None):
    return sum([np.prod(p.size()) for p in module.parameters()])


class FunctionalParamVectorWrapper(nn.Module):
    """
    This wraps a module so that it takes params in the forward pass
    """

    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        _functional, self.named_params = make_functional(module)
        self.functional = [_functional]  # remove params from being counted

    def forward(self, param_vector: torch.Tensor, *args, **kwargs):
        params = []
        start = 0
        for p in self.named_params:
            end = start + np.prod(p.size())
            params.append(param_vector[start:end].view(p.size()))
            start = end
        return self.functional[0](params, *args, **kwargs)
