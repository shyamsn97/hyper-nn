from __future__ import annotations

from typing import Any, Dict, List, Optional

import flax.linen as nn
import jax.numpy as jnp

from hypernn.base import EmbeddingModule
from hypernn.jax.utils import count_jax_params


class FlaxEmbeddingModule(nn.Module, EmbeddingModule):
    embedding_dim: int
    num_embeddings: int
    target_input_shape: Optional[Any] = None

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
    ):
        return count_jax_params(target, target_input_shape, inputs=inputs)

    def __call__(self, inp: Optional[Any] = None) -> Dict[str, jnp.array]:
        raise NotImplementedError("__call__ nore implemented!")


class DefaultFlaxEmbeddingModule(FlaxEmbeddingModule):
    def setup(self):
        self.embedding = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self, inp: Optional[Any] = None) -> Dict[str, jnp.array]:
        indices = jnp.arange(0, self.num_embeddings)
        return {"embedding": self.embedding(indices)}
