from __future__ import annotations

import abc
import math
from collections.abc import Iterable
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


class EmbeddingModule(metaclass=abc.ABCMeta):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        target_input_shape: Optional[Any] = None,
    ):
        pass

    @classmethod
    @abc.abstractmethod
    def count_params(
        cls,
        target: Union[torch.nn.Module, flax.linen.Module],
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
    ):
        pass

    @classmethod
    def from_target(
        cls,
        target: Union[torch.nn.Module, flax.linen.Module],
        embedding_dim: int,
        num_embeddings: int,
        num_target_parameters: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> EmbeddingModule:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(target, target_input_shape, inputs)
        if hidden_dim is None:
            hidden_dim = math.ceil(num_target_parameters / num_embeddings)
            if hidden_dim != 0:
                remainder = num_target_parameters % hidden_dim
                if remainder > 0:
                    diff = math.ceil(remainder / hidden_dim)
                    num_embeddings += diff
        return cls(embedding_dim, num_embeddings, target_input_shape, *args, **kwargs)


class WeightGenerator(metaclass=abc.ABCMeta):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dim: int,
        target_input_shape: Optional[Any] = None,
    ):
        pass

    @classmethod
    @abc.abstractmethod
    def count_params(
        cls,
        target: Union[torch.nn.Module, flax.linen.Module],
        target_input_shape: Optional[Any] = None,
        *args,
        **kwargs
    ):
        pass

    @classmethod
    def from_target(
        cls,
        target: Union[torch.nn.Module, flax.linen.Module],
        embedding_dim: int,
        num_embeddings: int,
        num_target_parameters: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        target_input_shape: Optional[Any] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ) -> WeightGenerator:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(target, target_input_shape, inputs)
        if hidden_dim is None:
            hidden_dim = math.ceil(num_target_parameters / num_embeddings)
            if hidden_dim != 0:
                remainder = num_target_parameters % hidden_dim
                if remainder > 0:
                    diff = math.ceil(remainder / hidden_dim)
                    num_embeddings += diff
        return cls(
            embedding_dim,
            num_embeddings,
            hidden_dim,
            target_input_shape,
            *args,
            **kwargs
        )


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
    def generate_params(
        self,
        inp: Iterable[Any] = [],
        embedding_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
    ) -> Any:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network
        """

    @abc.abstractmethod
    def forward(
        self,
        generated_params: Optional[Union[torch.tensor, jnp.array]] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
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
