from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Any, List, Optional

import torch
import torch.nn as nn

from hypernn.base import WeightGenerator
from hypernn.torch.utils import count_params


class TorchWeightGenerator(nn.Module, WeightGenerator, metaclass=abc.ABCMeta):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dim: int,
        target_input_shape: Optional[Any] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.target_input_shape = target_input_shape
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
    ):
        return count_params(target, target_input_shape, inputs=inputs)

    @abc.abstractmethod
    def forward(
        self, embedding: torch.Tensor, inp: Iterable[Any] = [], *args, **kwargs
    ) -> torch.Tensor:
        """
        Generate Embedding
        """

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device


class DefaultTorchWeightGenerator(TorchWeightGenerator):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dim: int,
        target_input_shape: Optional[Any] = None,
    ):
        super().__init__(embedding_dim, num_embeddings, hidden_dim, target_input_shape)
        self.generator = nn.Linear(embedding_dim, hidden_dim)

    def forward(
        self, embedding: torch.Tensor, inp: Iterable[Any] = [], *args, **kwargs
    ) -> torch.Tensor:
        return self.generator(embedding).view(-1)
