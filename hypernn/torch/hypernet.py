import copy
import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from hypernn.torch.embedding_module import StaticEmbeddingModule, TorchEmbeddingModule

# internal
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params
from hypernn.torch.weight_generator import LinearWeightGenerator, TorchWeightGenerator


class TorchHyperNetwork(nn.Module):
    def __init__(
        self,
        target_network: nn.Module,
        embedding_module_constructor: Callable[
            [int, int], TorchEmbeddingModule
        ] = StaticEmbeddingModule,
        weight_generator_constructor: Callable[
            [int, int], TorchWeightGenerator
        ] = LinearWeightGenerator,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
    ):
        super(TorchHyperNetwork, self).__init__()
        self.__device_param_dummy__ = nn.Parameter(torch.empty(0))
        self.num_parameters = count_params(target_network)
        self.embedding_module_constructor = embedding_module_constructor
        self.weight_generator_constructor = weight_generator_constructor
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.hidden_dim = hidden_dim
        if self.hidden_dim is None:
            self.hidden_dim = math.ceil(self.num_parameters / self.num_embeddings)
            if self.hidden_dim != 0:
                remainder = self.num_parameters % self.hidden_dim
                if remainder > 0:
                    diff = math.ceil(remainder / self.hidden_dim)
                    self.num_embeddings += diff

        self.embedding_module = embedding_module_constructor(
            self.embedding_dim, self.num_embeddings
        )
        self.weight_generator = weight_generator_constructor(
            self.embedding_dim, self.hidden_dim
        )

        self.target = self.create_functional_target_network(
            copy.deepcopy(target_network)
        )

    def create_functional_target_network(self, target_network: nn.Module):
        return FunctionalParamVectorWrapper(target_network)

    def generate_params(self, *args, **kwargs):
        embeddings = self.embedding_module()
        params = self.weight_generator(embeddings).view(-1)
        return params

    def forward(self, x, params=None):
        if params is None:
            params = self.generate_params(x)
        return self.target(params, x)

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
