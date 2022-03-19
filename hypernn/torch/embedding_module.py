from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn


class TorchEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self, embedding_dim: int, num_embeddings: int, input_shape: Optional[Any] = None
    ):
        super().__init__()
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = None
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    @classmethod
    def from_target(
        cls,
        target: nn.Module,
        embedding_dim: int,
        num_embeddings: int,
        input_shape: Optional[Any] = None,
        *args,
        **kwargs
    ) -> TorchEmbeddingModule:
        return cls(embedding_dim, num_embeddings, input_shape, *args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    @abc.abstractmethod
    def forward(self, inp: Iterable[Any] = [], *args, **kwargs) -> torch.Tensor:
        """
        Generate Embedding
        """


class DefaultTorchEmbeddingModule(TorchEmbeddingModule):
    def __init__(
        self, embedding_dim: int, num_embeddings: int, input_shape: Optional[Any] = None
    ):
        super().__init__(embedding_dim, num_embeddings, input_shape)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inp: Iterable[Any] = [], *args, **kwargs) -> torch.Tensor:
        indices = torch.arange(self.num_embeddings).to(self.device)
        return self.embedding(indices)
