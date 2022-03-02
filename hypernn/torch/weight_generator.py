from __future__ import annotations

import abc

import torch
import torch.nn as nn


class TorchWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

    @abc.abstractmethod
    def forward(self, embedding: torch.Tensor, *args, **kwargs):
        """
        Generate Embedding
        """


class LinearWeightGenerator(TorchWeightGenerator):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__(embedding_dim, hidden_dim)
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim, bias=False),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.generator(embedding).view(-1)
