import abc
from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp


class FlaxWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    embedding_dim: int
    hidden_dim: int

    @abc.abstractmethod
    def __call__(
        self, embedding: jnp.array, inp: Optional[Any] = None, *args, **kwargs
    ):
        """
        Forward pass to output embeddings
        """


class DefaultFlaxWeightGenerator(FlaxWeightGenerator):
    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim)

    def __call__(self, embedding: jnp.array, inp: Optional[Any] = None):
        return self.dense1(embedding)
