from __future__ import annotations

import abc
import math
from typing import Any, Dict, List, Optional, Tuple

import flax
import flax.linen as nn
import jax.numpy as jnp

from hypernn.jax.utils import count_jax_params


@abc.abstractmethod
class FlaxEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
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

    @classmethod
    def from_target(
        cls,
        target: flax.linen.Module,
        embedding_dim: int,
        num_embeddings: int,
        num_target_parameters: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> FlaxEmbeddingModule:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(target, target_input_shape, inputs)
        if hidden_dim is None:
            hidden_dim = math.ceil(num_target_parameters / num_embeddings)
            if hidden_dim != 0:
                remainder = num_target_parameters % hidden_dim
                if remainder > 0:
                    diff = math.ceil(remainder / hidden_dim)
                    num_embeddings += diff
        return cls(embedding_dim, num_embeddings, target_input_shape, *args, **kwargs)

    def __call__(
        self, inp: Optional[Any] = None, *args, **kwargs
    ) -> Tuple[jnp.array, Dict[str, jnp.array]]:
        raise NotImplementedError("__call__ not implemented!")


class DefaultFlaxEmbeddingModule(FlaxEmbeddingModule):
    def setup(self):
        self.embedding = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(
        self, inp: Optional[Any] = None
    ) -> Tuple[jnp.array, Dict[str, jnp.array]]:
        indices = jnp.arange(0, self.num_embeddings)
        return self.embedding(indices), {}
