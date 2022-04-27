from __future__ import annotations

import abc
import copy
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa

import torch
import torch.nn as nn

from hypernn.base import HyperNetwork
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params


class BaseTorchHyperNetwork(nn.Module, HyperNetwork, metaclass=abc.ABCMeta):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: int,
    ):
        super().__init__()
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

        self._target = self.create_functional_target_network(
            copy.deepcopy(target_network)
        )

        self.num_target_parameters = num_target_parameters

    def create_functional_target_network(self, target_network: nn.Module):
        func_model = FunctionalParamVectorWrapper(target_network)
        return func_model

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[Any] = None,
        num_target_parameters: Optional[int] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> BaseTorchHyperNetwork:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(
                target_network, target_input_shape, inputs=inputs
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            *args,
            **kwargs
        )

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[Any] = None,
        return_variables: bool = False,
    ):
        return count_params(target, target_input_shape, inputs=inputs)

    @abc.abstractmethod
    def generate_params(
        self, inp: Iterable[Any], *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass

    def forward(
        self,
        inp: Iterable[Any] = [],
        generated_params: Optional[torch.Tensor] = None,
        has_aux: bool = False,
        *args,
        **kwargs
    ):
        aux_output = {}
        if generated_params is None:
            generated_params, aux_output = self.generate_params(inp, *args, **kwargs)

        if has_aux:
            return self._target(generated_params, *inp), generated_params, aux_output
        return self._target(generated_params, *inp)

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
