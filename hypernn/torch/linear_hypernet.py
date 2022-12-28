from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple  # noqa

import torch
import torch.nn as nn
from functorch import vmap  # noqa

from hypernn.torch.hypernet import TorchHyperNetwork
from hypernn.torch.utils import get_weight_chunk_dims


class TorchLinearHyperNetwork(TorchHyperNetwork):
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
        if weight_chunk_dim is None:
            self.weight_chunk_dim = get_weight_chunk_dims(
                self.num_target_parameters, num_embeddings
            )

        self.custom_embedding_module = custom_embedding_module
        self.custom_weight_generator = custom_weight_generator
        self.setup()

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

    def generate_params(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding_module(
            torch.arange(self.num_embeddings, device=self.device)
        )
        generated_params = self.weight_generator(embedding).view(-1)
        return generated_params, {"embedding": embedding}
