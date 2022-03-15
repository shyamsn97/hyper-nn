import abc
from typing import Any, Callable, Optional, Union

import flax
import jax.numpy as jnp
import torch

from hypernn.jax.embedding_module import FlaxEmbeddingModule
from hypernn.jax.weight_generator import FlaxWeightGenerator
from hypernn.torch.embedding_module import TorchEmbeddingModule
from hypernn.torch.weight_generator import TorchWeightGenerator

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


class BaseHyperNetwork(metaclass=abc.ABCMeta):
    def __init__(
        self,
        target_network: Union[torch.nn.Module, flax.linen.Module],
        embedding_module_constructor: Callable[
            [int, int], Union[TorchEmbeddingModule, FlaxEmbeddingModule]
        ],
        weight_generator_constructor: Callable[
            [int, int], Union[TorchWeightGenerator, FlaxWeightGenerator]
        ],
        embedding_dim: int,
        num_embeddings: int,
    ):
        pass

    @abc.abstractmethod
    def generate_params(self, inp: Optional[Any] = None, *args, **kwargs) -> Any:
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
        inp: Any,
        params: Optional[Union[torch.tensor, jnp.array]] = None,
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
