from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from hypernn.jax.hypernet import BaseFlaxHyperNetwork
from hypernn.jax.utils import get_hidden_weight_generator_dims


class FlaxHyperNetwork(BaseFlaxHyperNetwork):
    embedding_dim: int = 100
    num_embeddings: int = 3
    hidden_dim: Optional[int] = None
    embedding_module: Optional[nn.Module] = None
    weight_generator_module: Optional[nn.Module] = None

    def setup(self):
        if self.embedding_module is None:
            self.embedding = self.make_embedding()
        else:
            self.embedding = self.embedding_module

        if self.weight_generator_module is None:
            self.weight_generator = self.make_weight_generator()
        else:
            self.weight_generator = self.weight_generator_module

    def make_embedding(self) -> nn.Module:
        return nn.Embed(self.num_embeddings, self.embedding_dim)

    def make_weight_generator(self) -> nn.Module:
        return nn.Dense(self.hidden_dim)

    def generate_params(
        self, inp: Iterable[Any] = []
    ) -> Tuple[jnp.array, Dict[str, Any]]:
        indices = jnp.arange(0, self.num_embeddings)
        embedding = self.embedding(indices)
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
        hidden_dim: Optional[int] = None,
        embedding_module: Optional[nn.Module] = None,
        weight_generator_module: Optional[nn.Module] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> FlaxHyperNetwork:
        num_target_parameters, variables = cls.count_params(
            target_network, target_input_shape, inputs=inputs, return_variables=True
        )
        _value_flat, target_treedef = jax.tree_util.tree_flatten(variables)
        target_weight_shapes = [v.shape for v in _value_flat]

        if hidden_dim is None:
            hidden_dim = get_hidden_weight_generator_dims(
                num_target_parameters, num_embeddings
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            target_treedef=target_treedef,
            target_weight_shapes=target_weight_shapes,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dim=hidden_dim,
            embedding_module=embedding_module,
            weight_generator_module=weight_generator_module,
            *args,
            **kwargs,
        )
