from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn

from hypernn.base import EmbeddingModule
from hypernn.torch.utils import count_params


class TorchEmbeddingModule(nn.Module, EmbeddingModule, metaclass=abc.ABCMeta):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        target_input_shape: Optional[Any] = None,
    ):
        super().__init__()
        self.target_input_shape = target_input_shape
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = None
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        return_variables: bool = False,
    ):
        return count_params(target, target_input_shape, return_variables)

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
        self,
        embedding_dim: int,
        num_embeddings: int,
        target_input_shape: Optional[Any] = None,
    ):
        super().__init__(embedding_dim, num_embeddings, target_input_shape)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inp: Iterable[Any] = [], *args, **kwargs) -> torch.Tensor:
        indices = torch.arange(self.num_embeddings).to(self.device)
        return self.embedding(indices)
