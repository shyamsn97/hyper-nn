import copy
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from hypernn.base import HyperNetwork
from hypernn.torch.embedding_module import (
    DefaultTorchEmbeddingModule,
    TorchEmbeddingModule,
)
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params
from hypernn.torch.weight_generator import (
    DefaultTorchWeightGenerator,
    TorchWeightGenerator,
)


class TorchHyperNetwork(nn.Module, HyperNetwork):

    DEFAULT_EMBEDDING_MODULE = DefaultTorchEmbeddingModule
    DEFAULT_WEIGHT_GENERATOR = DefaultTorchWeightGenerator

    def __init__(
        self,
        target_input_shape: Any,
        target_network: nn.Module,
        embedding_module: Optional[TorchEmbeddingModule] = None,
        weight_generator: Optional[TorchWeightGenerator] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
    ):
        super(TorchHyperNetwork, self).__init__()
        self.target_input_shape = target_input_shape
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device
        self.embedding_module = embedding_module
        self.weight_generator = weight_generator

        if self.embedding_module is None:
            num_target_parameters = self.count_params(
                target_network, self.target_input_shape
            )
            self.embedding_module = self.DEFAULT_EMBEDDING_MODULE.from_target(
                target_network,
                embedding_dim,
                num_embeddings,
                num_target_parameters=num_target_parameters,
                target_input_shape=self.target_input_shape,
            )

        if self.weight_generator is None:
            self.weight_generator = self.DEFAULT_WEIGHT_GENERATOR.from_target(
                target_network,
                self.embedding_module.embedding_dim,
                self.embedding_module.num_embeddings,
                num_target_parameters=num_target_parameters,
                hidden_dim=hidden_dim,
                target_input_shape=self.target_input_shape,
            )

        self._target = self.create_functional_target_network(
            copy.deepcopy(target_network)
        )

    def create_functional_target_network(self, target_network: nn.Module):
        func_model = FunctionalParamVectorWrapper(target_network)
        return func_model

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        return_variables: bool = False,
    ):
        return count_params(target, target_input_shape, return_variables)

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
