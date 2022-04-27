from __future__ import annotations

import abc
from dataclasses import field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax._src.tree_util import PyTreeDef

from hypernn.base import HyperNetwork
from hypernn.jax.utils import count_jax_params


def target_forward(apply_fn, param_tree, *args, **kwargs):
    return apply_fn(param_tree, *args, **kwargs)


class BaseFlaxHyperNetwork(nn.Module, HyperNetwork, metaclass=abc.ABCMeta):
    target_network: nn.Module
    num_target_parameters: int
    target_treedef: PyTreeDef
    target_weight_shapes: List[Any] = field(default_factory=list)

    def create_param_tree(self, generated_params):
        param_list = []
        curr = 0
        for shape in self.target_weight_shapes:
            num_params = np.prod(shape)
            param_list.append(generated_params[curr : curr + num_params].reshape(shape))
            curr = curr + num_params

        param_tree = jax.tree_util.tree_unflatten(self.target_treedef, param_list)
        return param_tree

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[List[Any]] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> BaseFlaxHyperNetwork:
        num_target_parameters, variables = cls.count_params(
            target_network, target_input_shape, inputs=inputs, return_variables=True
        )
        _value_flat, target_treedef = jax.tree_util.tree_flatten(variables)
        target_weight_shapes = [v.shape for v in _value_flat]
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            target_treef=target_treedef,
            target_weight_shapes=target_weight_shapes,
            *args,
            **kwargs
        )

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
        return_variables: bool = False,
    ):
        return count_jax_params(
            target, target_input_shape, inputs=inputs, return_variables=return_variables
        )

    @abc.abstractmethod
    def generate_params(
        self, inp: Iterable[Any], *args, **kwargs
    ) -> Tuple[jnp.array, Dict[str, Any]]:
        pass

    def forward(
        self,
        inp: Iterable[Any] = [],
        generated_params: Optional[jnp.array] = None,
        has_aux: bool = True,
        *args,
        **kwargs
    ) -> Tuple[jnp.array, List[jnp.array]]:
        aux_output = {}
        if generated_params is None:
            generated_params, aux_output = self.generate_params(inp, *args, **kwargs)

        param_tree = self.create_param_tree(generated_params)

        if has_aux:
            return (
                target_forward(self.target_network.apply, param_tree, *inp),
                generated_params,
                aux_output,
            )
        return target_forward(self.target_network.apply, param_tree, *inp)

    def __call__(
        self,
        inp: Iterable[Any] = [],
        generated_params: Optional[jnp.array] = None,
        has_aux: bool = True,
        *args,
        **kwargs
    ) -> Tuple[jnp.array, List[jnp.array]]:
        return self.forward(inp, generated_params, has_aux, *args, **kwargs)

    def save(self, params, path: str):
        bytes_output = serialization.to_bytes(params)
        with open(path, "wb") as f:
            f.write(bytes_output)

    def load(self, params, path: str):
        with open(path, "rb") as f:
            bytes_output = f.read()
        return serialization.from_bytes(params, bytes_output)
