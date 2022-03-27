from typing import Any, List, Optional

import flax.linen as nn
import jax.numpy as jnp

from hypernn.base import WeightGenerator
from hypernn.jax.utils import count_jax_params


class FlaxWeightGenerator(nn.Module, WeightGenerator):
    embedding_dim: int
    num_embeddings: int
    hidden_dim: int
    target_input_shape: Optional[Any] = None

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
    ):
        return count_jax_params(target, target_input_shape, inputs=inputs)


class DefaultFlaxWeightGenerator(FlaxWeightGenerator):
    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim)

    def __call__(self, embedding: jnp.array, inp: Optional[Any] = None):
        return self.dense1(embedding)
