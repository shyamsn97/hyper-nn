from typing import List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def count_jax_params(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    inputs: Optional[List[jnp.array]] = None,
    return_variables: bool = False,
) -> int:
    if input_shape is None and inputs is None:
        raise ValueError("Input shape or inputs must be specified")
    if inputs is None:
        inputs = jnp.zeros(input_shape)
    variables = jax.lax.stop_gradient(model.init(jax.random.PRNGKey(0), inputs))

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
