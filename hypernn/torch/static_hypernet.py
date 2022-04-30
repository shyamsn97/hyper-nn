from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa

import torch
import torch.nn as nn

from hypernn.torch.hypernet import BaseTorchHyperNetwork
from hypernn.torch.utils import get_hidden_weight_generator_dims


class TorchHyperNetwork(BaseTorchHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
        embedding_module: Optional[nn.Module] = None,
        weight_generator_module: Optional[nn.Module] = None,
    ):
        super().__init__(target_network, num_target_parameters)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim

        self.embedding = embedding_module
        self.weight_generator = weight_generator_module

        if self.embedding is None:
            self.embedding = self.make_embedding()

        if self.weight_generator is None:
            self.weight_generator = self.make_weight_generator()

    def make_embedding(self) -> nn.Module:
        return nn.Embedding(self.num_embeddings, self.embedding_dim)

    def make_weight_generator(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.hidden_dim)

    def generate_params(
        self, inp: Iterable[Any] = []
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding(torch.arange(self.num_embeddings).to(self.device))
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
        hidden_dim: Optional[int] = None,
        embedding_module: Optional[nn.Module] = None,
        weight_generator_module: Optional[nn.Module] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> TorchHyperNetwork:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(
                target_network, target_input_shape, inputs=inputs
            )
        if hidden_dim is None:
            hidden_dim = get_hidden_weight_generator_dims(
                num_target_parameters, num_embeddings
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dim=hidden_dim,
            embedding_module=embedding_module,
            weight_generator_module=weight_generator_module,
            *args,
            **kwargs,
        )
