from __future__ import annotations

from dataclasses import field
from typing import Any, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax._src.tree_util import PyTreeDef

from hypernn.base import HyperNetwork
from hypernn.jax.embedding_module import DefaultFlaxEmbeddingModule, FlaxEmbeddingModule
from hypernn.jax.utils import count_jax_params
from hypernn.jax.weight_generator import DefaultFlaxWeightGenerator, FlaxWeightGenerator


class _MetaFlaxHyperNetwork(type):
    def __call__(
        cls: FlaxHyperNetwork,
        target_input_shape: Any,
        target_network: nn.Module,
        embedding_module: Optional[FlaxEmbeddingModule] = None,
        weight_generator: Optional[FlaxWeightGenerator] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
        *args,
        **kwargs
    ) -> FlaxHyperNetwork:
        num_target_parameters, variables = count_jax_params(
            target_network, target_input_shape, return_variables=True
        )
        _value_flat, target_treedef = jax.tree_util.tree_flatten(variables)
        target_weight_shapes = [v.shape for v in _value_flat]

        if embedding_module is None:
            embedding_module = DefaultFlaxEmbeddingModule.from_target(
                target_network,
                embedding_dim,
                num_embeddings,
                num_target_parameters=num_target_parameters,
                target_input_shape=target_input_shape,
            )

        if weight_generator is None:
            weight_generator = DefaultFlaxWeightGenerator.from_target(
                target_network,
                embedding_module.embedding_dim,
                embedding_module.num_embeddings,
                num_target_parameters=num_target_parameters,
                hidden_dim=hidden_dim,
                target_input_shape=target_input_shape,
            )

        instance = cls.__new__(
            cls
        )  # __new__ is actually a static method - cls has to be passed explicitly
        if isinstance(instance, cls):
            instance.__init__(
                target_input_shape=target_input_shape,
                target_network=target_network,
                target_treedef=target_treedef,
                embedding_module=embedding_module,
                weight_generator=weight_generator,
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                hidden_dim=hidden_dim,
                target_weight_shapes=target_weight_shapes,
            )
        return instance


def target_forward(apply_fn, param_tree, inputs):
    return apply_fn(param_tree, inputs)


class FlaxHyperNetwork(nn.Module, HyperNetwork, metaclass=_MetaFlaxHyperNetwork):
    target_input_shape: Any
    target_network: nn.Module
    target_treedef: PyTreeDef
    embedding_module: FlaxEmbeddingModule
    weight_generator: FlaxWeightGenerator
    embedding_dim: int
    num_embeddings: int
    hidden_dim: int
    target_weight_shapes: Optional[List[Any]] = field(default_factory=list)

    def setup(self):
        pass

    @classmethod
    def count_params(cls, target: nn.Module, target_input_shape: Optional[Any] = None):
        return count_jax_params(target, target_input_shape)

    def generate_params(
        self, x: Optional[Any] = None, *args, **kwargs
    ) -> List[jnp.array]:
        embeddings = self.embedding_module(x)
        params = self.weight_generator(embeddings, x).reshape(-1)
        param_list = []
        curr = 0
        for shape in self.target_weight_shapes:
            num_params = np.prod(shape)
            param_list.append(params[curr : curr + num_params].reshape(shape))
            curr = curr + num_params
        return param_list, embeddings

    def forward(
        self, inp: Any, params: Optional[List[jnp.array]] = None, *args, **kwargs
    ):
        if params is None:
            params, embeddings = self.generate_params(inp, *args, **kwargs)
        param_tree = jax.tree_util.tree_unflatten(self.target_treedef, params)
        return (
            target_forward(param_tree, inp),
            params,
            embeddings,
        )

    def __call__(
        self, inp: Any, params: Optional[List[jnp.array]] = None, *args, **kwargs
    ) -> Tuple[jnp.array, List[jnp.array]]:
        return self.forward(inp, params, *args, **kwargs)

    def save(self, params, path: str):
        bytes_output = serialization.to_bytes(params)
        with open(path, "wb") as f:
            f.write(bytes_output)

    def load(self, params, path: str):
        with open(path, "rb") as f:
            bytes_output = f.read()
        return serialization.from_bytes(params, bytes_output)
