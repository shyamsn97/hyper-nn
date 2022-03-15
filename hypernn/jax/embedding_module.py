import abc

import flax.linen as nn
import jax.numpy as jnp


class FlaxEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    embedding_dim: int
    num_embeddings: int

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Forward pass to output embeddings
        """


class DefaultFlaxEmbeddingModule(FlaxEmbeddingModule):
    def setup(self):
        self.embedding = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self):
        indices = jnp.arange(0, self.num_embeddings)
        return self.embedding(indices)
