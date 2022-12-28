from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax._src.tree_util import PyTreeDef

from hypernn.base import HyperNetwork
from hypernn.jax.utils import count_jax_params


def create_param_tree(generated_params, target_weight_shapes, target_treedef):
    param_list = []
    curr = 0
    for shape in target_weight_shapes:
        num_params = np.prod(shape)
        param_list.append(generated_params[curr : curr + num_params].reshape(shape))
        curr = curr + num_params

    param_tree = jax.tree_util.tree_unflatten(target_treedef, param_list)
    return param_tree


class JaxHyperNetwork(nn.Module, HyperNetwork):
    target_network: nn.Module
    num_target_parameters: int
    target_treedef: PyTreeDef
    target_weight_shapes: List[Any] = field(default_factory=list)

    def assert_parameter_shapes(self, generated_params):
        assert generated_params.shape[-1] >= self.num_target_parameters

    def generate_params(self, *args, **kwargs) -> Tuple[jnp.array, Dict[str, Any]]:
        raise NotImplementedError("generate_params not implemented!")

    def target_forward(
        self,
        *args,
        generated_params: jnp.array,
        assert_parameter_shapes: bool = True,
        **kwargs,
    ) -> jnp.array:

        if assert_parameter_shapes:
            self.assert_parameter_shapes(generated_params)

        param_tree = create_param_tree(
            generated_params, self.target_weight_shapes, self.target_treedef
        )

        return self.target_network.apply(param_tree, *args, **kwargs)

    def forward(
        self,
        *args,
        generated_params: Optional[jnp.array] = None,
        has_aux: bool = False,
        assert_parameter_shapes: bool = True,
        generate_params_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> Tuple[jnp.array, List[jnp.array]]:
        """
        Main method for creating / using generated parameters and passing in input into the target network

        Args:
            generated_params (Optional[jnp.array], optional): Generated parameters of the target network. If not provided, the hypernetwork will generate the parameters. Defaults to None.
            has_aux (bool, optional): If True, return the auxiliary output from generate_params method. Defaults to False.
            assert_parameter_shapes (bool, optional): If True, raise an error if generated_params does not have shape (num_target_parameters,). Defaults to True.
            generate_params_kwargs (Dict[str, Any], optional): kwargs to be passed to generate_params method

        Returns:
            output (torch.Tensor) | (jnp.array, Dict[str, jnp.array]): returns output from target network and optionally auxiliary output.
        """
        aux_output = {}
        if generated_params is None:
            generated_params, aux_output = self.generate_params(
                **generate_params_kwargs
            )

        if has_aux:
            return (
                self.target_forward(
                    *args,
                    generated_params=generated_params,
                    assert_parameter_shapes=assert_parameter_shapes,
                    **kwargs,
                ),
                generated_params,
                aux_output,
            )
        return self.target_forward(
            *args,
            generated_params=generated_params,
            assert_parameter_shapes=assert_parameter_shapes,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        generated_params: Optional[jnp.array] = None,
        has_aux: bool = False,
        assert_parameter_shapes: bool = True,
        **kwargs,
    ) -> Tuple[jnp.array, List[jnp.array]]:
        return self.forward(
            *args,
            generated_params=generated_params,
            has_aux=has_aux,
            assert_parameter_shapes=assert_parameter_shapes,
            **kwargs,
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

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[List[Any]] = None,
        num_target_parameters: Optional[int] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> JaxHyperNetwork:
        num_target_parameters, variables = cls.count_params(
            target_network, target_input_shape, inputs=inputs, return_variables=True
        )
        _value_flat, target_treedef = jax.tree_util.tree_flatten(variables)
        target_weight_shapes = [v.shape for v in _value_flat]

        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            target_treedef=target_treedef,
            target_weight_shapes=target_weight_shapes,
            *args,
            **kwargs,
        )

    def save(self, params, path: str):
        bytes_output = serialization.to_bytes(params)
        with open(path, "wb") as f:
            f.write(bytes_output)

    def load(self, params, path: str):
        with open(path, "rb") as f:
            bytes_output = f.read()
        return serialization.from_bytes(params, bytes_output)
