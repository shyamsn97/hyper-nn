import math
from typing import List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def get_weight_chunk_dims(num_target_parameters: int, num_embeddings: int):
    weight_chunk_dim = math.ceil(num_target_parameters / num_embeddings)
    if weight_chunk_dim != 0:
        remainder = num_target_parameters % weight_chunk_dim
        if remainder > 0:
            diff = math.ceil(remainder / weight_chunk_dim)
            num_embeddings += diff
    return weight_chunk_dim


def count_jax_params(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    inputs: Optional[List[jnp.array]] = None,
    return_variables: bool = False,
) -> int:
    if input_shape is None and inputs is None:
        raise ValueError("Input shape or inputs must be specified")
    if inputs is None:
        inputs = [jnp.zeros(shape) for shape in input_shape]
    variables = model.init(jax.random.PRNGKey(0), *inputs)

    def count_recursive(d):
        s = 0
        if isinstance(d, int):
            return d
        for k in d:
            s += count_recursive(d[k])
        return s

    param_counts = jax.tree_map(lambda x: int(np.prod(x.shape)), variables)["params"]
    if return_variables:
        return count_recursive(param_counts), variables
    return count_recursive(param_counts)
