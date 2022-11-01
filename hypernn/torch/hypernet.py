from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple  # noqa

import torch
import torch.nn as nn

from hypernn.base import HyperNetwork
from hypernn.torch.utils import (
    FunctionalParamVectorWrapper,
    count_params,
    get_weight_chunk_dims,
)


def create_functional_target_network(target_network: nn.Module):
    func_model = FunctionalParamVectorWrapper(target_network)
    return func_model


class TorchHyperNetwork(nn.Module, HyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
        custom_embedding_module: Optional[nn.Module] = None,
        custom_weight_generator: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.target_network = create_functional_target_network(
            copy.deepcopy(target_network)
        )
        self.target_weight_shapes = self.target_network.target_weight_shapes

        self.num_target_parameters = num_target_parameters

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.weight_chunk_dim = weight_chunk_dim
        self.custom_embedding_module = custom_embedding_module
        self.custom_weight_generator = custom_weight_generator
        self.setup()

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    def setup(self):
        if self.custom_embedding_module is None:
            self.embedding_module = self.make_embedding_module()
        else:
            self.embedding_module = self.custom_embedding_module

        if self.custom_weight_generator is None:
            self.weight_generator = self.make_weight_generator()
        else:
            self.weight_generator = self.custom_weight_generator_module

    def assert_parameter_shapes(self, generated_params):
        assert generated_params.shape[-1] >= self.num_target_parameters

    def make_embedding_module(self) -> nn.Module:
        return nn.Embedding(self.num_embeddings, self.embedding_dim)

    def make_weight_generator(self) -> nn.Module:
        return nn.Linear(self.embedding_dim, self.weight_chunk_dim)

    def generate_params(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding_module(
            torch.arange(self.num_embeddings, device=self.device)
        )
        generated_params = self.weight_generator(embedding).view(-1)
        return generated_params, {"embedding": embedding}

    def target_forward(
        self,
        *args,
        generated_params: torch.Tensor,
        assert_parameter_shapes: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if assert_parameter_shapes:
            self.assert_parameter_shapes(generated_params)

        return self.target_network(generated_params, *args, **kwargs)

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
                *args, **kwargs, **generate_params_kwargs
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
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> TorchHyperNetwork:
        if num_target_parameters is None:
            num_target_parameters = cls.count_params(
                target_network, target_input_shape, inputs=inputs
            )
        if weight_chunk_dim is None:
            weight_chunk_dim = get_weight_chunk_dims(
                num_target_parameters, num_embeddings
            )
        return cls(
            target_network=target_network,
            num_target_parameters=num_target_parameters,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            weight_chunk_dim=weight_chunk_dim,
            *args,
            **kwargs,
        )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
