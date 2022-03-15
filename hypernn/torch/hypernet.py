import copy
import math
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

from hypernn.base_hypernet import BaseHyperNetwork
from hypernn.torch.embedding_module import (
    DefaultTorchEmbeddingModule,
    TorchEmbeddingModule,
)
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params
from hypernn.torch.weight_generator import (
    DefaultTorchWeightGenerator,
    TorchWeightGenerator,
)

# from functorch import make_functional


class TorchHyperNetwork(nn.Module, BaseHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        embedding_module_constructor: Callable[
            [int, int], TorchEmbeddingModule
        ] = DefaultTorchEmbeddingModule,
        weight_generator_constructor: Callable[
            [int, int], TorchWeightGenerator
        ] = DefaultTorchWeightGenerator,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
    ):
        super(TorchHyperNetwork, self).__init__()
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device
        self.num_parameters = count_params(target_network)
        self.embedding_module_constructor = embedding_module_constructor
        self.weight_generator_constructor = weight_generator_constructor
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.setup_dims()

        self.embedding_module, self.weight_generator = self.get_networks()
        self._target = self.create_functional_target_network(
            copy.deepcopy(target_network)
        )

    def create_functional_target_network(self, target_network: nn.Module):
        return FunctionalParamVectorWrapper(target_network)

    def setup_dims(self):
        if self.hidden_dim is None:
            self.hidden_dim = math.ceil(self.num_parameters / self.num_embeddings)
            if self.hidden_dim != 0:
                remainder = self.num_parameters % self.hidden_dim
                if remainder > 0:
                    diff = math.ceil(remainder / self.hidden_dim)
                    self.num_embeddings += diff

    def get_networks(self) -> Tuple[TorchEmbeddingModule, TorchWeightGenerator]:
        embedding_module = self.embedding_module_constructor(
            self.embedding_dim, self.num_embeddings
        )
        weight_generator = self.weight_generator_constructor(
            self.embedding_dim, self.hidden_dim
        )
        return embedding_module, weight_generator

    def generate_params(
        self, inp: Optional[Any] = None, *args, **kwargs
    ) -> torch.Tensor:
        embeddings = self.embedding_module(inp)
        params = self.weight_generator(embeddings, inp).view(-1)
        return params

    def forward(
        self, inp: Any, params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if params is None:
            params = self.generate_params(inp)
        return self._target(params, inp), params

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
