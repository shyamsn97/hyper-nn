import abc

import flax.linen as nn
import jax.numpy as jnp


class FlaxWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    embedding_dim: int
    hidden_dim: int

    @abc.abstractmethod
    def __call__(self, embedding: jnp.array, *args, **kwargs):
        """
        Forward pass to output embeddings
        """


class FlaxStaticWeightGenerator(FlaxWeightGenerator):
    def setup(self):
        self.dense1 = nn.Dense(32)
        self.dense2 = nn.Dense(self.hidden_dim)

    def __call__(self, embedding: jnp.array):
        x = self.dense1(embedding)
        x = nn.relu(x)
        x = self.dense2(x)
        return x
