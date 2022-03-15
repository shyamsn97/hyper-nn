import abc
from typing import Any, Optional

"""
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
"""


class BaseHyperNetwork(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_params(self, inp: Optional[Any] = None, *args, **kwargs) -> Any:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network
        """
