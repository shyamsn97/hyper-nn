from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Union

import flax
import jax.numpy as jnp
import torch

"""
                            Static HyperNetwork
                                                      ┌───────────┐
                                                      │   Input   │
                                                      └─────┬─────┘
                                                            │
                                                            ▼
┌────────────────────────────────────────────┐  ┌────────────────────────┐
│                                            │  │                        │
│              HyperNetwork                  │  │      Target Network    │
│ ┌───────────┐          ┌─────────────────┐ │  │  ┌─────────────────┐   │
│ │           │          │                 │ │  │  │                 │   │
│ │ Embedding ├─────────►│ Weight Generator├─┼──┼─►│Generated Weights│   │
│ │           │          │                 │ │  │  │                 │   │
│ └───────────┘          └─────────────────┘ │  │  └─────────────────┘   │
│                                            │  │                        │
└────────────────────────────────────────────┘  └───────────┬────────────┘
                                                            │
                                                            │
                                                            ▼
                                                      ┌───────────┐
                                                      │   Output  │
                                                      └───────────┘

                            Dynamic Hypernetwork
                                                          ┌───────────┐
            ┌───────────────── and/or ┌───────────────────┤   Input   │
            │                         │                   └─────┬─────┘
            │                         │                         │
            │                         │                         ▼
    ┌───────┼─────────────────────────┼──────────┐  ┌───────────────────────┐
    │       │                         │          │  │                       │
    │       │      HyperNetwork       │          │  │     Target Network    │
    │ ┌─────▼─────┐          ┌────────▼────────┐ │  │  ┌─────────────────┐  │
    │ │           │          │                 │ │  │  │                 │  │
    │ │ Embedding ├─────────►│ Weight Generator├─┼──┼─►│Generated Weights│  │
    │ │  Module   │          │                 │ │  │  │                 │  │
    │ └───────────┘          └─────────────────┘ │  │  └─────────────────┘  │
    │                                            │  │                       │
    └────────────────────────────────────────────┘  └───────────┬───────────┘
                                                                │
                                                                ▼
                                                          ┌───────────┐
                                                          │   Output  │
                                                          └───────────┘

"""


class BaseModule(metaclass=abc.ABCMeta):
    @classmethod
    def from_target(
        cls,
        target: Union[torch.nn.Module, flax.linen.Module],
        num_target_parameters: Optional[int] = None,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> BaseModule:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(target, target_input_shape, inputs)
        return cls(num_target_parameters, *args, **kwargs)


class HyperNetwork(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def count_params(
        cls,
        target: Union[torch.nn.Module, flax.linen.Module],
        target_input_shape: Optional[Any] = None,
    ):
        """
        Counts parameters of target nn.Module

        Args:
            target (Union[torch.nn.Module, flax.linen.Module]): _description_
            target_input_shape (Optional[Any], optional): _description_. Defaults to None.
        """

    @classmethod
    @abc.abstractmethod
    def from_target(
        cls, target: Union[torch.nn.Module, flax.linen.Module], *args, **kwargs
    ) -> HyperNetwork:
        """
        creates hypernetwork from target

        Args:
            cls (_type_): _description_
        """

    @abc.abstractmethod
    def forward(
        self,
        generated_params: Optional[Union[torch.tensor, jnp.array]] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
        has_aux: bool = True,
        *args,
        **kwargs
    ):
        """
        Computes a forward pass with generated parameters or with parameters that are passed in

        Args:
            inp (Any): input from system
            params (Optional[Union[torch.tensor, jnp.array]], optional): Generated params. Defaults to None.
        Returns:
            returns output and generated parameters
        """
