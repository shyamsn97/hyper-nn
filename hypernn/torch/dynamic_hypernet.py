from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa

import torch
import torch.nn as nn

from hypernn.torch.hypernet import TorchHyperNetwork


class DynamicHyperNetwork(TorchHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        target_input_shape: Optional[List[Any]] = None,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
        embedding_module: Optional[nn.Module] = None,
        weight_generator_module: Optional[nn.Module] = None,
    ):
        super().__init__(target_network, target_input_shape, num_target_parameters)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim

        self.embedding_module = embedding_module
        self.weight_generator_module = weight_generator_module

    def default_embedding_module(self) -> nn.Module:
        return nn.Embedding(self.num_embeddings, self.embedding_dim)

    def default_weight_generator_module(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.hidden_dim)

    def embedding(self, inp: Iterable[Any] = []) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.embedding_module is None:
            self.embedding_module = self.default_embedding_module()
        return self.embedding_module(torch.arange(self.num_embeddings).to(self.device))

    def weight_generator(
        self, embedding: torch.Tensor, inp: Iterable[Any] = []
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.weight_generator_module is None:
            self.weight_generator_module = self.default_weight_generator_module()
        return self.weight_generator_module(embedding).view(-1)

    def generate_params(
        self, inp: Iterable[Any] = []
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding(inp)
        generated_params = self.weight_generator(embedding, inp)
        return generated_params, {"embedding": embedding}
