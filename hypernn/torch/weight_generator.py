from __future__ import annotations

import abc
import math
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from hypernn.torch.utils import count_params


class TorchWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
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

    @classmethod
    def from_target(
        cls,
        target: torch.nn.Module,
        embedding_dim: int,
        num_embeddings: int,
        num_target_parameters: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> TorchWeightGenerator:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(target, target_input_shape, inputs)
        if hidden_dim is None:
            hidden_dim = math.ceil(num_target_parameters / num_embeddings)
            if hidden_dim != 0:
                remainder = num_target_parameters % hidden_dim
                if remainder > 0:
                    diff = math.ceil(remainder / hidden_dim)
                    num_embeddings += diff
        return cls(
            embedding_dim,
            num_embeddings,
            hidden_dim,
            target_input_shape,
            *args,
            **kwargs
        )

    @abc.abstractmethod
    def forward(
        self, embedding: torch.Tensor, inp: Iterable[Any] = [], *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
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
        self,
        embedding: torch.Tensor,
        inp: Iterable[Any] = [],
    ) -> Dict[str, torch.Tensor]:
        return self.generator(embedding).view(-1), {}
