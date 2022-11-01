from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from hypernn.jax.hypernet import JaxHyperNetwork


class RNNCell(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, hidden_state):
        concatenated = jnp.concatenate((x, hidden_state), axis=-1)
        hidden_state = nn.Dense(self.hidden_dim)(concatenated)
        return nn.tanh(hidden_state)


class JaxDynamicEmbeddingModule(nn.Module):
    input_dim: int
    embedding_dim: int
    num_embeddings: int

    def setup(self):
        self.embedding = nn.Embed(
            self.num_embeddings,
            self.embedding_dim,
            embedding_init=jax.nn.initializers.uniform(),
        )
        self.rnn = RNNCell(self.num_embeddings)

    def init_hidden(self):
        return jnp.zeros((1, self.num_embeddings))

    def __call__(self, x: jnp.array, hidden_state: Optional[jnp.array] = None):
        if hidden_state is None:
            hidden_state = self.init_hidden()
        indices = jnp.arange(0, self.num_embeddings)
        hidden_state = self.rnn(x, hidden_state)
        embedding = self.embedding(indices) * hidden_state.reshape(
            self.num_embeddings, 1
        )
        return embedding, hidden_state


class JaxDynamicHyperNetwork(JaxHyperNetwork):
    input_dim: int = 0

    def make_embedding_module(self):
        return JaxDynamicEmbeddingModule(
            self.input_dim, self.embedding_dim, self.num_embeddings
        )

    def generate_params(
        self, *args, hidden_state: Optional[jnp.array] = None, **kwargs
    ) -> Tuple[jnp.array, Dict[str, Any]]:
        embedding, hidden_state = self.embedding_module(
            *args, **kwargs, hidden_state=hidden_state
        )
        generated_params = self.weight_generator(embedding).reshape(-1)
        return generated_params, {"embedding": embedding, "hidden_state": hidden_state}
