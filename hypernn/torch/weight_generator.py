from __future__ import annotations

import abc
from typing import Any, Optional

import torch
import torch.nn as nn


class TorchWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.__device_param_dummy__ = nn.Parameter(torch.empty(0))

    @abc.abstractmethod
    def forward(
        self, embedding: torch.Tensor, inp: Optional[Any] = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Generate Embedding
        """

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device


class DefaultTorchWeightGenerator(TorchWeightGenerator):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__(embedding_dim, hidden_dim)
        self.generator = nn.Linear(embedding_dim, hidden_dim)

    def forward(
        self, embedding: torch.Tensor, inp: Optional[Any] = None
    ) -> torch.Tensor:
        return self.generator(embedding).view(-1)
