from typing import Callable, Optional, Tuple, Any, List
import math
import jax

from hypernn.base_hypernet import BaseHyperNetwork
from hypernn.jax.embedding_module import FlaxEmbeddingModule, FlaxStaticEmbeddingModule
from hypernn.jax.weight_generator import FlaxWeightGenerator, FlaxStaticWeightGenerator
from hypernn.jax.utils import count_jax_params

def FlaxHyperNetwork(
    input_shape: Tuple[int, ...],
    target_network: nn.Module,
    embedding_module_constructor: Callable[[int, int], FlaxEmbeddingModule] = EmbeddingModule,
    weight_generator_constructor: Callable[[int, int], FlaxWeightGenerator] = StaticWeightGenerator,
    embedding_dim: int = 100,
    num_embeddings: int = 3,
    hidden_dim: Optional[int] = None
):
    class FlaxHyperNetwork(nn.Module, BaseHyperNetwork):
        _target: nn.Module
        embedding_module_constructor: Callable[[int, int], FlaxEmbeddingModule] = FlaxStaticEmbeddingModule
        weight_generator_constructor: Callable[[int, int], FlaxWeightGenerator] = FlaxStaticWeightGenerator
        embedding_dim: int = 100

        def setup(self):
            self.num_parameters, variables = count_jax_params(self._target, input_shape)
            self.setup_dims()
            self.embedding_module, self.weight_generator = self.get_networks()

            _value_flat, self.target_treedef = jax.tree_util.tree_flatten(variables)
            self.target_weight_shapes = [v.shape for v in _value_flat]

        @nn.nowrap
        def setup_dims(self):
            self.num_embeddings = num_embeddings
            self.hidden_dim = hidden_dim
            if self.hidden_dim is None:
                self.hidden_dim = math.ceil(self.num_parameters / self.num_embeddings)
                if self.hidden_dim != 0:
                    remainder = self.num_parameters % self.hidden_dim
                    if remainder > 0:
                        diff = math.ceil(remainder / self.hidden_dim)
                        self.num_embeddings += diff

        @nn.nowrap
        def get_networks(self) -> Tuple[FlaxEmbeddingModule, FlaxWeightGenerator]:
            embedding_module = self.embedding_module_constructor(
                self.embedding_dim, self.num_embeddings
            )
            weight_generator = self.weight_generator_constructor(
                self.embedding_dim, self.hidden_dim
            )
            return embedding_module, weight_generator

        def generate_params(self, x: Optional[Any] = None, *args, **kwargs) -> List[jnp.array]:
            embeddings = self.embedding_module()
            params = self.weight_generator(embeddings).reshape(-1)
            param_list = []
            curr = 0
            for shape in self.target_weight_shapes:
                num_params = np.prod(shape)
                param_list.append(params[curr:curr+num_params].reshape(shape))
                curr = curr+num_params
            return param_list

        def __call__(self, x: Any, params: Optional[List[jnp.array]] = None) -> Tuple[jnp.array, List[jnp.array]]:
            if params is None:
                params = self.generate_params(x)
            param_tree = jax.tree_util.tree_unflatten(self.target_treedef, params)
            return self._target.apply(param_tree, x), params

    return FlaxHyperNetwork(target_network, embedding_module_constructor, weight_generator_constructor, embedding_dim)
