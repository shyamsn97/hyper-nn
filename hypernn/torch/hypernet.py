import copy
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from hypernn.base_hypernet import BaseHyperNetwork
from hypernn.torch.embedding_module import (
    DefaultTorchEmbeddingModule,
    TorchEmbeddingModule,
)
from hypernn.torch.utils import FunctionalParamVectorWrapper
from hypernn.torch.weight_generator import (
    DefaultTorchWeightGenerator,
    TorchWeightGenerator,
)


class TorchHyperNetwork(nn.Module, BaseHyperNetwork):

    DEFAULT_EMBEDDING_MODULE = DefaultTorchEmbeddingModule
    DEFAULT_WEIGHT_GENERATOR = DefaultTorchWeightGenerator

    def __init__(
        self,
        input_shape: Any,
        target_network: nn.Module,
        embedding_module: Optional[TorchEmbeddingModule] = None,
        weight_generator: Optional[TorchWeightGenerator] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
    ):
        super(TorchHyperNetwork, self).__init__()
        self.input_shape = input_shape
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device
        self.embedding_module = embedding_module
        self.weight_generator = weight_generator

        if self.embedding_module is None:
            self.embedding_module = self.DEFAULT_EMBEDDING_MODULE(
                embedding_dim, num_embeddings, self.input_shape
            )

        if self.weight_generator is None:
            self.weight_generator = self.DEFAULT_WEIGHT_GENERATOR.from_target(
                self.target_network,
                self.embedding_module.embedding_dim,
                self.embedding_module.num_embeddings,
                hidden_dim,
            )

        self._target = self.create_functional_target_network(
            copy.deepcopy(target_network)
        )

    def create_functional_target_network(self, target_network: nn.Module):
        func_model = FunctionalParamVectorWrapper(target_network)
        return func_model

    def generate_params(
        self,
        inp: Iterable[Any] = [],
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
    ) -> torch.Tensor:
        embedding_output = self.embedding_module(inp, **embedding_module_kwargs)
        params = self.weight_generator(
            embedding_output, inp, **weight_generator_kwargs
        ).view(-1)
        return params, embedding_output

    def forward(
        self,
        inp: Iterable[Any] = [],
        generated_params: Optional[torch.Tensor] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        embedding_output = None
        if generated_params is None:
            generated_params, embedding_output = self.generate_params(
                inp, embedding_module_kwargs, weight_generator_kwargs
            )
        return self._target(generated_params, *inp), generated_params, embedding_output

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
