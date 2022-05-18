from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa

import torch
import torch.nn as nn

from hypernn.torch.hypernet import BaseTorchHyperNetwork
from hypernn.torch.utils import get_hidden_weight_generator_dims


class DynamicEmbeddingModule(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.rnn_cell = nn.RNNCell(self.input_dim, self.num_embeddings)

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros((1, self.num_embeddings), device=self.device)

    def forward(
        self,
        inp: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        if hidden_state is None:
            hidden_state = self.init_hidden()
        hidden_state = self.rnn_cell(inp, hidden_state)
        indices = torch.arange(self.num_embeddings).to(self.device)
        embedding = self.embedding(indices) * hidden_state.view(self.num_embeddings, 1)
        return embedding, hidden_state


class DynamicHyperNetwork(BaseTorchHyperNetwork):
    def __init__(
        self,
        input_dim: int,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
        embedding_module: Optional[nn.Module] = None,
        weight_generator_module: Optional[nn.Module] = None,
    ):
        super().__init__(target_network, num_target_parameters)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.embedding_module = embedding_module
        self.weight_generator_module = weight_generator_module
        self.setup()

    def setup(self) -> None:
        if self.embedding_module is None:
            self.embedding_module = DynamicEmbeddingModule(
                self.input_dim, self.embedding_dim, self.num_embeddings
            )

        if self.weight_generator_module is None:
            self.weight_generator_module = nn.Linear(
                self.embedding_dim, self.hidden_dim
            )

    def embedding(
        self, inp: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embedding, hidden_state = self.embedding_module(inp, hidden_state)
        return embedding, hidden_state

    def weight_generator(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.weight_generator_module(embedding).view(-1)

    def generate_params(
        self, inp: Iterable[Any] = [], hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding, hidden_state = self.embedding(*inp, hidden_state=hidden_state)
        generated_params = self.weight_generator(embedding)
        return generated_params, {"embedding": embedding, "hidden_state": hidden_state}

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
    ) -> DynamicHyperNetwork:
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
