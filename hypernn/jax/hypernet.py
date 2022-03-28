from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

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


def target_forward(apply_fn, param_tree, *args, **kwargs):
    return apply_fn(param_tree, *args, **kwargs)


class FlaxHyperNetwork(nn.Module, HyperNetwork):
    target_network: nn.Module
    target_input_shape: List[Any]
    target_treedef: PyTreeDef
    embedding_module: Optional[
        Union[FlaxEmbeddingModule, Type[FlaxEmbeddingModule]]
    ] = None
    weight_generator: Optional[
        Union[FlaxWeightGenerator, Type[FlaxWeightGenerator]]
    ] = None
    embedding_dim: int = 100
    num_embeddings: int = 3
    hidden_dim: Optional[int] = None
    num_target_parameters: Optional[int] = None
    embedding_module_kwargs: Dict[str, Any] = field(default_factory=lambda: ({}))
    weight_generator_kwargs: Dict[str, Any] = field(default_factory=lambda: ({}))
    target_weight_shapes: List[Any] = field(default_factory=list)

    def setup(self):
        num_target_parameters = self.num_target_parameters
        if num_target_parameters is None:
            num_target_parameters = self.count_params(
                self.target_network, self.target_input_shape
            )

        embedding_module = self.embedding_module
        if embedding_module is None:
            embedding_module = DefaultFlaxEmbeddingModule

        weight_generator = self.weight_generator
        if weight_generator is None:
            weight_generator = DefaultFlaxWeightGenerator

        self._embedding_module = embedding_module.from_target(
            target=self.target_network,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            num_target_parameters=num_target_parameters,
            hidden_dim=self.hidden_dim,
            target_input_shape=self.target_input_shape,
            **self.embedding_module_kwargs
        )

        self._weight_generator = weight_generator.from_target(
            target=self.target_network,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            num_target_parameters=num_target_parameters,
            hidden_dim=self.hidden_dim,
            target_input_shape=self.target_input_shape,
            **self.weight_generator_kwargs
        )

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[List[Any]] = None,
        embedding_module: Optional[
            Union[FlaxEmbeddingModule, Type[FlaxEmbeddingModule]]
        ] = None,
        weight_generator: Optional[
            Union[FlaxWeightGenerator, Type[FlaxWeightGenerator]]
        ] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
        num_target_parameters: Optional[int] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> FlaxHyperNetwork:
        num_target_parameters, variables = count_jax_params(
            target_network, target_input_shape, inputs=inputs, return_variables=True
        )
        _value_flat, target_treedef = jax.tree_util.tree_flatten(variables)
        target_weight_shapes = [v.shape for v in _value_flat]
        return cls(
            target_network=target_network,
            target_input_shape=target_input_shape,
            target_treedef=target_treedef,
            embedding_module=embedding_module,
            weight_generator=weight_generator,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dim=hidden_dim,
            num_target_parameters=num_target_parameters,
            target_weight_shapes=target_weight_shapes,
            embedding_module_kwargs=embedding_module_kwargs,
            weight_generator_kwargs=weight_generator_kwargs,
            *args,
            **kwargs
        )

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
    ):
        return count_jax_params(target, target_input_shape, inputs=inputs)

    def create_param_tree(self, generated_params):
        param_list = []
        curr = 0
        for shape in self.target_weight_shapes:
            num_params = np.prod(shape)
            param_list.append(generated_params[curr : curr + num_params].reshape(shape))
            curr = curr + num_params

        param_tree = jax.tree_util.tree_unflatten(self.target_treedef, param_list)
        return param_tree

    def generate_params(
        self,
        inp: Iterable[Any] = [],
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network
        """
        embedding_module_output = self._embedding_module(inp, **embedding_module_kwargs)
        assert (
            isinstance(embedding_module_output, dict)
            and "embedding" in embedding_module_output
        )

        weight_generator_output = self._weight_generator(
            embedding_module_output, inp, **weight_generator_kwargs
        )

        assert (
            isinstance(weight_generator_output, dict)
            and "params" in weight_generator_output
        )

        return (
            weight_generator_output["params"],
            embedding_module_output,
            weight_generator_output,
        )

    def forward(
        self,
        inp: List[Any],
        generated_params: Optional[jnp.array] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
        has_aux: bool = True,
    ):
        embedding_module_output = None
        weight_generator_output = None

        if generated_params is None:
            (
                generated_params,
                embedding_module_output,
                weight_generator_output,
            ) = self.generate_params(
                inp, embedding_module_kwargs, weight_generator_kwargs
            )

        param_tree = self.create_param_tree(generated_params)

        if not has_aux:
            return target_forward(self.target_network.apply, param_tree, *inp)

        return (
            target_forward(self.target_network.apply, param_tree, *inp),
            generated_params,
            embedding_module_output,
            weight_generator_output,
        )

    def __call__(
        self,
        inp: List[Any],
        generated_params: Optional[jnp.array] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
        has_aux: bool = True,
    ) -> Tuple[jnp.array, List[jnp.array]]:
        return self.forward(
            inp,
            generated_params,
            embedding_module_kwargs,
            weight_generator_kwargs,
            has_aux,
        )

    def save(self, params, path: str):
        bytes_output = serialization.to_bytes(params)
        with open(path, "wb") as f:
            f.write(bytes_output)

    def load(self, params, path: str):
        with open(path, "rb") as f:
            bytes_output = f.read()
        return serialization.from_bytes(params, bytes_output)
