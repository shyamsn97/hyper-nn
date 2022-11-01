from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

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
    │ │           │          │                 │ │  │  │                 │  │
    │ └───────────┘          └─────────────────┘ │  │  └─────────────────┘  │
    │                                            │  │                       │
    └────────────────────────────────────────────┘  └───────────┬───────────┘
                                                                │
                                                                ▼
                                                          ┌───────────┐
                                                          │   Output  │
                                                          └───────────┘

"""


class HyperNetwork(metaclass=abc.ABCMeta):
    def setup(self) -> None:
        self.embedding_module = self.make_embedding_module()
        self.weight_generator = self.make_weight_generator()

    @abc.abstractmethod
    def make_embedding_module(self):
        """
        Makes an embedding module to be used

        Returns:
            a torch.nn.Module or flax.linen.Module that can be used to return an embedding matrix to be used to generate weights
        """

    @abc.abstractmethod
    def make_weight_generator(self):
        """
        Makes an embedding module to be used

        Returns:
            a torch.nn.Module or flax.linen.Module that can be used to return an embedding matrix to be used to generate weights
        """

    @classmethod
    @abc.abstractmethod
    def count_params(
        cls,
        target,
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
    def from_target(cls, target, *args, **kwargs) -> HyperNetwork:
        """
        creates hypernetwork from target

        Args:
            cls (_type_): _description_
        """

    @abc.abstractmethod
    def generate_params(
        self, inp: Optional[Any] = None, *args, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network and a dictionary of extra info
        """

    @abc.abstractmethod
    def forward(
        self,
        *args,
        generated_params=None,
        has_aux: bool = True,
        **kwargs,
    ):
        """
        Computes a forward pass with generated parameters or with parameters that are passed in

        Args:
            inp (Any): input from system
            generated_params (Optional[Union[torch.tensor, jnp.array]], optional): Generated params. Defaults to None.
            has_aux (bool): flag to indicate whether to return auxiliary info
        Returns:
            returns output and generated params and auxiliary info if has_aux is provided
        """
