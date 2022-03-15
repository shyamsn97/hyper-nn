import math
from dataclasses import field
from typing import Any, Callable, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.tree_util import PyTreeDef
from flax import serialization

from hypernn.base_hypernet import BaseHyperNetwork
from hypernn.jax.embedding_module import DefaultFlaxEmbeddingModule, FlaxEmbeddingModule
from hypernn.jax.utils import count_jax_params
from hypernn.jax.weight_generator import DefaultFlaxWeightGenerator, FlaxWeightGenerator


def target_forward(apply_fn, param_tree, inputs):
    return apply_fn(param_tree, inputs)


def FlaxHyperNetwork(
    input_shape: Tuple[int, ...],
    target_network: nn.Module,
    embedding_module_constructor: Callable[
        [int, int], FlaxEmbeddingModule
    ] = DefaultFlaxEmbeddingModule,
    weight_generator_constructor: Callable[
        [int, int], FlaxWeightGenerator
    ] = DefaultFlaxWeightGenerator,
    embedding_dim: int = 100,
    num_embeddings: int = 3,
    hidden_dim: Optional[int] = None,
):
    num_parameters, variables = count_jax_params(target_network, input_shape)
    _value_flat, target_treedef = jax.tree_util.tree_flatten(variables)
    target_weight_shapes = [v.shape for v in _value_flat]

    num_embeddings = num_embeddings
    hidden_dim = hidden_dim
    if hidden_dim is None:
        hidden_dim = math.ceil(num_parameters / num_embeddings)
        if hidden_dim != 0:
            remainder = num_parameters % hidden_dim
            if remainder > 0:
                diff = math.ceil(remainder / hidden_dim)
                num_embeddings += diff

    class FlaxHyperNetwork(nn.Module, BaseHyperNetwork):
        _target: nn.Module
        target_treedef: PyTreeDef
        num_parameters: int
        num_embeddings: int
        hidden_dim: int
        embedding_module_constructor: Callable[
            [int, int], FlaxEmbeddingModule
        ] = DefaultFlaxEmbeddingModule
        weight_generator_constructor: Callable[
            [int, int], FlaxWeightGenerator
        ] = DefaultFlaxWeightGenerator
        embedding_dim: int = 100
        target_weight_shapes: Optional[List[Any]] = field(default_factory=list)

        def setup(self):
            self.embedding_module, self.weight_generator = self.get_networks()

        def get_networks(self) -> Tuple[FlaxEmbeddingModule, FlaxWeightGenerator]:
            embedding_module = self.embedding_module_constructor(
                self.embedding_dim, self.num_embeddings
            )
            weight_generator = self.weight_generator_constructor(
                self.embedding_dim, self.hidden_dim
            )
            return embedding_module, weight_generator

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
            return param_list

        def __call__(
            self, x: Any, params: Optional[List[jnp.array]] = None
        ) -> Tuple[jnp.array, List[jnp.array]]:
            if params is None:
                params = self.generate_params(x)
            param_tree = jax.tree_util.tree_unflatten(self.target_treedef, params)
            return target_forward(self._target.apply, param_tree, x), params

        def save(self, params, path: str):
            bytes_output = serialization.to_bytes(params)
            with open(path, 'wb') as f: 
                f.write(bytes_output)

        def load(self, params, path: str):
            with open(path, "rb") as f:
                bytes_output = f.read()
            return serialization.from_bytes(params, bytes_output)

    return FlaxHyperNetwork(
        target_network,
        target_treedef,
        num_parameters,
        num_embeddings,
        hidden_dim,
        embedding_module_constructor,
        weight_generator_constructor,
        embedding_dim,
        target_weight_shapes,
    )
