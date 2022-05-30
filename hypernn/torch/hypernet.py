from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa

import torch
import torch.nn as nn

from hypernn.base import HyperNetwork
from hypernn.torch.utils import (
    FunctionalParamVectorWrapper,
    count_params,
    get_weight_chunk_dims,
)


def create_functional_target_network(target_network: nn.Module):
    func_model = FunctionalParamVectorWrapper(target_network)
    return func_model


class TorchHyperNetwork(nn.Module, HyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
        custom_embedding_module: Optional[nn.Module] = None,
        custom_weight_generator: Optional[nn.Module] = None,
    ):
        super().__init__()

        self._target = create_functional_target_network(copy.deepcopy(target_network))
        self.target_weight_shapes = self._target.target_weight_shapes

        self.num_target_parameters = num_target_parameters

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.weight_chunk_dim = weight_chunk_dim
        self.custom_embedding_module = custom_embedding_module
        self.custom_weight_generator = custom_weight_generator
        self.setup()

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    def setup(self):
        if self.custom_embedding_module is None:
            self.embedding_module = self.make_embedding_module()
        else:
            self.embedding_module = self.custom_embedding_module

        if self.custom_weight_generator is None:
            self.weight_generator = self.make_weight_generator()
        else:
            self.weight_generator = self.custom_weight_generator_module

    def make_embedding_module(self) -> nn.Module:
        return nn.Embedding(self.num_embeddings, self.embedding_dim)

    def make_weight_generator(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.weight_chunk_dim)

    def assert_parameter_shapes(self, generated_params):
        assert generated_params.shape[-1] >= self.num_target_parameters

    def generate_params(
        self, inp: Iterable[Any] = []
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding_module(
            torch.arange(self.num_embeddings, device=self.device)
        )
        generated_params = self.weight_generator(embedding).view(-1)
        return generated_params, {"embedding": embedding}

    def forward(
        self,
        inp: Iterable[Any] = [],
        generated_params: Optional[torch.Tensor] = None,
        has_aux: bool = False,
        *args,
        **kwargs,
    ):
        aux_output = {}
        if generated_params is None:
            generated_params, aux_output = self.generate_params(inp, *args, **kwargs)

        self.assert_parameter_shapes(generated_params)

        if has_aux:
            return self._target(generated_params, *inp), generated_params, aux_output
        return self._target(generated_params, *inp)

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[Any] = None,
    ):
        return count_params(target, target_input_shape, inputs=inputs)

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[Any] = None,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> TorchHyperNetwork:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(
                target_network, target_input_shape, inputs=inputs
            )
        if weight_chunk_dim is None:
            weight_chunk_dim = get_weight_chunk_dims(
                num_target_parameters, num_embeddings
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            weight_chunk_dim=weight_chunk_dim,
            *args,
            **kwargs,
        )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
