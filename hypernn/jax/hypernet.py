from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from hypernn.jax.base import BaseFlaxHyperNetwork
from hypernn.jax.utils import get_weight_chunk_dims


class EmbeddingModule(nn.Module):
    embedding_dim: int
    num_embeddings: int

    def setup(self):
        self.embedding = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self):
        return self.embedding(jnp.arange(0, self.num_embeddings))


class JaxHyperNetwork(BaseFlaxHyperNetwork):
    embedding_dim: int = 100
    num_embeddings: int = 3
    weight_chunk_dim: Optional[int] = None
    custom_embedding_module: Optional[nn.Module] = None
    custom_weight_generator: Optional[nn.Module] = None

    def setup(self):
        if self.custom_embedding_module is None:
            self.embedding_module = self.make_embedding_module()
        else:
            self.embedding_module = self.custom_embedding_module

        if self.custom_weight_generator is None:
            self.weight_generator = self.make_weight_generator()
        else:
            self.weight_generator = self.custom_weight_generator_module

    def make_embedding_module(self):
        return EmbeddingModule(self.embedding_dim, self.num_embeddings)

    def make_weight_generator(self):
        return nn.Dense(self.weight_chunk_dim)

    def generate_params(
        self, inp: Iterable[Any] = []
    ) -> Tuple[jnp.array, Dict[str, Any]]:
        embedding = self.embedding_module()
        generated_params = self.weight_generator(embedding).reshape(-1)
        return generated_params, {"embedding": embedding}

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[List[Any]] = None,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> JaxHyperNetwork:
        num_target_parameters, variables = cls.count_params(
            target_network, target_input_shape, inputs=inputs, return_variables=True
        )
        _value_flat, target_treedef = jax.tree_util.tree_flatten(variables)
        target_weight_shapes = [v.shape for v in _value_flat]

        if weight_chunk_dim is None:
            weight_chunk_dim = get_weight_chunk_dims(
                num_target_parameters, num_embeddings
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            target_treedef=target_treedef,
            target_weight_shapes=target_weight_shapes,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            weight_chunk_dim=weight_chunk_dim,
            *args,
            **kwargs,
        )
