from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa

import torch
import torch.nn as nn

from hypernn.torch.base import BaseTorchHyperNetwork
from hypernn.torch.utils import get_weight_chunk_dims


class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    def forward(self) -> torch.Tensor:
        return self.embedding(torch.arange(self.num_embeddings, device=self.device))

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device


class TorchHyperNetwork(BaseTorchHyperNetwork):
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
        super().__init__(target_network, num_target_parameters)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.weight_chunk_dim = weight_chunk_dim
        self.embedding_module = custom_embedding_module
        self.weight_generator = custom_weight_generator
        self.setup()

    def setup(self) -> None:
        if self.embedding_module is None:
            self.embedding_module = self.make_embedding_module()

        if self.weight_generator is None:
            self.weight_generator = self.make_weight_generator()

    def make_embedding_module(self) -> nn.Module:
        return EmbeddingModule(self.num_embeddings, self.embedding_dim)

    def make_weight_generator(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.weight_chunk_dim)

    def generate_params(
        self, inp: Iterable[Any] = []
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding_module()
        generated_params = self.weight_generator(embedding).view(-1)
        return generated_params, {"embedding": embedding}

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
