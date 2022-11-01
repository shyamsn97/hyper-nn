from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, Union  # noqa

import torch
import torch.nn as nn

from hypernn.torch.hypernet import TorchHyperNetwork


class TorchDynamicEmbeddingModule(nn.Module):
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
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        if hidden_state is None:
            hidden_state = self.init_hidden()

        hidden_state = self.rnn_cell(x, hidden_state)
        indices = torch.arange(self.num_embeddings, device=self.device)
        embedding = self.embedding(indices) * hidden_state.view(self.num_embeddings, 1)
        return embedding, hidden_state


class TorchDynamicHyperNetwork(TorchHyperNetwork):
    def __init__(
        self,
        input_dim: int,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
        custom_embedding_module: Optional[nn.Module] = None,
        custom_weight_generator: Optional[nn.Module] = None,
    ):
        self.input_dim = input_dim

        super().__init__(
            target_network,
            num_target_parameters,
            embedding_dim,
            num_embeddings,
            weight_chunk_dim,
            custom_embedding_module,
            custom_weight_generator,
        )

    def make_embedding_module(self) -> nn.Module:
        return TorchDynamicEmbeddingModule(
            self.input_dim, self.embedding_dim, self.num_embeddings
        )

    def make_weight_generator(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.weight_chunk_dim)

    def generate_params(
        self, *args, hidden_state: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding, hidden_state = self.embedding_module(
            *args, **kwargs, hidden_state=hidden_state
        )
        generated_params = self.weight_generator(embedding).view(-1)
        return generated_params, {"embedding": embedding, "hidden_state": hidden_state}
