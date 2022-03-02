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

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Generate Embedding
        """


class StaticEmbeddingModule(TorchEmbeddingModule):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__(embedding_dim, num_embeddings)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, *args, **kwargs):
        indices = torch.arange(self.num_embeddings)
        return self.embedding(indices)
