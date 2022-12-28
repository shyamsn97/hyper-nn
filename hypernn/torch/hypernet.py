from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple  # noqa

import torch
import torch.nn as nn
from functorch import vmap  # noqa

from hypernn.base import HyperNetwork
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params


def create_functional_target_network(target_network: nn.Module):
    func_model = FunctionalParamVectorWrapper(target_network)
    return func_model


class TorchHyperNetwork(nn.Module, HyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
    ):
        super().__init__()

        self.functional_target_network = create_functional_target_network(
            target_network
        )
        self.target_weight_shapes = self.functional_target_network.target_weight_shapes

        self.num_target_parameters = num_target_parameters
        if num_target_parameters is None:
            self.num_target_parameters = count_params(target_network)

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    def assert_parameter_shapes(self, generated_params):
        assert generated_params.shape[-1] >= self.num_target_parameters

    def generate_params(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError("Generate params not implemented!")

    def target_forward(
        self,
        *args,
        generated_params: torch.Tensor,
        assert_parameter_shapes: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if assert_parameter_shapes:
            self.assert_parameter_shapes(generated_params)

        return self.functional_target_network(generated_params, *args, **kwargs)

    def forward(
        self,
        *args,
        generated_params: Optional[torch.Tensor] = None,
        has_aux: bool = False,
        assert_parameter_shapes: bool = True,
        generate_params_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """
        Main method for creating / using generated parameters and passing in input into the target network

        Args:
            generated_params (Optional[torch.Tensor], optional): Generated parameters of the target network. If not provided, the hypernetwork will generate the parameters. Defaults to None.
            has_aux (bool, optional): If True, return the auxiliary output from generate_params method. Defaults to False.
            assert_parameter_shapes (bool, optional): If True, raise an error if generated_params does not have shape (num_target_parameters,). Defaults to True.
            generate_params_kwargs (Dict[str, Any], optional): kwargs to be passed to generate_params method
            *args, *kwargs, arguments to be passed into the target network (also gets passed into generate_params)
        Returns:
            output (torch.Tensor) | (torch.Tensor, Dict[str, torch.Tensor]): returns output from target network and optionally auxiliary output.
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

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[Any] = None,
    ):
        return count_params(target, target_input_shape, inputs=inputs)

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[Any] = None,
        num_target_parameters: Optional[int] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> TorchHyperNetwork:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(
                target_network, target_input_shape, inputs=inputs
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            *args,
            **kwargs,
        )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
