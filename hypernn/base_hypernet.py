import abc
from typing import Any, Optional, Tuple


class BaseHyperNetwork(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def setup_dims(self) -> None:
        """
        Sets up hidden dimensions and num_embeddings based on target_network number of parameters
        """

    @abc.abstractmethod
    def get_networks(self) -> Tuple[Any, Any]:
        """
        Outputs Embedding Module and Weight Generator

        Returns:
            Tuple[Any, Any]: Embedding Module and Weight Generator
        """

    @abc.abstractmethod
    def generate_params(self, inp: Optional[Any] = None, *args, **kwargs) -> Any:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network
        """
