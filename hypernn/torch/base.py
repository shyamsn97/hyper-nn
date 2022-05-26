# from __future__ import annotations

# import abc
# import copy
# from collections.abc import Iterable
# from typing import Any, Dict, List, Optional, Tuple, Type, Union  # noqa

# import torch
# import torch.nn as nn

# from hypernn.base import HyperNetwork
# from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params


# class BaseTorchHyperNetwork(nn.Module, HyperNetwork, metaclass=abc.ABCMeta):
#     def __init__(self):
#         super().__init__()

#     @abc.abstractmethod
#     @classmethod
#     def from_target(
#         cls,
#         *args,
#         **kwargs,
#     ) -> BaseTorchHyperNetwork:
#         pass

#     @abc.abstractmethod
#     def generate_params(
#         self, inp: Iterable[Any], *args, **kwargs
#     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
#         pass

#     @abc.abstractmethod
#     def forward(
#         self,
#         inp: Iterable[Any] = [],
#         generated_params: Optional[torch.Tensor] = None,
#         has_aux: bool = False,
#         *args,
#         **kwargs,
#     ):
#         pass
