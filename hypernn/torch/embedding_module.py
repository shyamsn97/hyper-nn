from __future__ import annotations

import abc

import torch
import torch.nn as nn


class TorchEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = None
        self.__device_param_dummy__ = nn.Parameter(torch.empty(0))

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Generate Embedding
        """

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device


class DefaultTorchEmbeddingModule(TorchEmbeddingModule):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__(embedding_dim, num_embeddings)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, *args, **kwargs):
        indices = torch.arange(self.num_embeddings).to(self.device)
        return self.embedding(indices)
