import copy
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from hypernn.base import HyperNetwork
from hypernn.torch.embedding_module import (
    DefaultTorchEmbeddingModule,
    TorchEmbeddingModule,
)
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params
from hypernn.torch.weight_generator import (
    DefaultTorchWeightGenerator,
    TorchWeightGenerator,
)


class TorchHyperNetwork(nn.Module, HyperNetwork):

    DEFAULT_EMBEDDING_MODULE = DefaultTorchEmbeddingModule
    DEFAULT_WEIGHT_GENERATOR = DefaultTorchWeightGenerator

    def __init__(
        self,
        target_network: nn.Module,
        target_input_shape: Optional[List[Any]] = None,
        embedding_module: Optional[
            Union[TorchEmbeddingModule, Type[TorchEmbeddingModule]]
        ] = None,
        weight_generator: Optional[
            Union[TorchWeightGenerator, Type[TorchWeightGenerator]]
        ] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
        num_target_parameters: Optional[int] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
    ):
        super(TorchHyperNetwork, self).__init__()
        self.target_input_shape = target_input_shape
        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device
        self.embedding_module = embedding_module
        self.weight_generator = weight_generator
        self.embedding_module_kwargs = embedding_module_kwargs
        self.weight_generator_kwargs = weight_generator_kwargs

        if num_target_parameters is None:
            num_target_parameters = self.count_params(
                target_network, self.target_input_shape
            )
        self.num_target_parameters = num_target_parameters

        if self.embedding_module is None:
            self.embedding_module = self.DEFAULT_EMBEDDING_MODULE

        if self.weight_generator is None:
            self.weight_generator = self.DEFAULT_WEIGHT_GENERATOR

        # class but not an instance of the class
        if not isinstance(self.embedding_module, TorchEmbeddingModule) and issubclass(
            self.embedding_module, TorchEmbeddingModule
        ):
            self.embedding_module = self.embedding_module.from_target(
                target_network,
                embedding_dim,
                num_embeddings,
                num_target_parameters=num_target_parameters,
                target_input_shape=self.target_input_shape,
                **embedding_module_kwargs
            )

        if not isinstance(self.weight_generator, TorchWeightGenerator) and issubclass(
            self.weight_generator, TorchWeightGenerator
        ):
            self.weight_generator = self.weight_generator.from_target(
                target_network,
                self.embedding_module.embedding_dim,
                self.embedding_module.num_embeddings,
                num_target_parameters=num_target_parameters,
                hidden_dim=hidden_dim,
                target_input_shape=self.target_input_shape,
                **weight_generator_kwargs
            )

        self._target = self.create_functional_target_network(
            copy.deepcopy(target_network)
        )

    @classmethod
    def from_target(
        cls,
        target_network: nn.Module,
        target_input_shape: Optional[Any] = None,
        embedding_module: Optional[
            Union[TorchEmbeddingModule, Type[TorchEmbeddingModule]]
        ] = None,
        weight_generator: Optional[
            Union[TorchWeightGenerator, Type[TorchWeightGenerator]]
        ] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
        inputs: Optional[List[Any]] = None,
        *args,
        **kwargs
    ):
        num_target_parameters = cls.count_params(
            target_network, target_input_shape, inputs=inputs
        )
        return cls(
            target_network=target_network,
            target_input_shape=target_input_shape,
            embedding_module=embedding_module,
            weight_generator=weight_generator,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dim=hidden_dim,
            num_target_parameters=num_target_parameters,
            embedding_module_kwargs=embedding_module_kwargs,
            weight_generator_kwargs=weight_generator_kwargs,
            *args,
            **kwargs
        )

    def create_functional_target_network(self, target_network: nn.Module):
        func_model = FunctionalParamVectorWrapper(target_network)
        return func_model

    @classmethod
    def count_params(
        cls,
        target: nn.Module,
        target_input_shape: Optional[Any] = None,
        return_variables: bool = False,
        inputs: Optional[Any] = None,
    ):
        return count_params(target, target_input_shape, inputs=inputs)

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
        embedding_module_output = self.embedding_module(inp, **embedding_module_kwargs)
        assert (
            isinstance(embedding_module_output, dict)
            and "embedding" in embedding_module_output
        )

        weight_generator_output = self.weight_generator(
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
        inp: Iterable[Any] = [],
        generated_params: Optional[torch.Tensor] = None,
        embedding_module_kwargs: Dict[str, Any] = {},
        weight_generator_kwargs: Dict[str, Any] = {},
        has_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:

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

        if not has_aux:
            return self._target(generated_params, *inp)

        return (
            self._target(generated_params, *inp),
            generated_params,
            embedding_module_output,
            weight_generator_output,
        )

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
