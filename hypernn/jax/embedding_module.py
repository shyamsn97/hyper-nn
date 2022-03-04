import abc
import flax.linen as nn
import jax

class FlaxEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    embedding_dim: int
    num_embeddings: int

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
            Forward pass to output embeddings
        """

class FlaxStaticEmbeddingModule(FlaxEmbeddingModule):

    def setup(self):
        self.embedding = nn.Dense( self.embedding_dim, use_bias=False)

    def __call__(self):
        indices = jax.nn.one_hot(jax.numpy.arange(0,self.num_embeddings), self.num_embeddings)
        return self.embedding(indices)
