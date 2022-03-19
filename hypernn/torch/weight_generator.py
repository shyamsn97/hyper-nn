from __future__ import annotations

import abc
import math
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn

from hypernn.torch.utils import count_params


class TorchWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dim: int,
        input_shape: Optional[Any] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    @abc.abstractmethod
    def forward(
        self, embedding: torch.Tensor, inp: Iterable[Any] = [], *args, **kwargs
    ) -> torch.Tensor:
        """
        Generate Embedding
        """

    @classmethod
    def from_target(
        cls,
        target: nn.Module,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dim: Optional[int] = None,
        input_shape: Optional[Any] = None,
        *args,
        **kwargs
    ) -> TorchWeightGenerator:
        num_target_parameters = count_params(target)
        if hidden_dim is None:
            hidden_dim = math.ceil(num_target_parameters / num_embeddings)
            if hidden_dim != 0:
                remainder = num_target_parameters % hidden_dim
                if remainder > 0:
                    diff = math.ceil(remainder / hidden_dim)
                    num_embeddings += diff
        return cls(
            embedding_dim, num_embeddings, hidden_dim, input_shape, *args, **kwargs
        )

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device


class DefaultTorchWeightGenerator(TorchWeightGenerator):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dim: int,
        input_shape: Optional[Any] = None,
    ):
        super().__init__(embedding_dim, num_embeddings, hidden_dim, input_shape)
        self.generator = nn.Linear(embedding_dim, hidden_dim)

    def forward(
        self, embedding: torch.Tensor, inp: Iterable[Any] = [], *args, **kwargs
    ) -> torch.Tensor:
        return self.generator(embedding).view(-1)
