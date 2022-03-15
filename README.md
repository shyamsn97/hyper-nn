# hyper-nn (Easy Hypernetworks in Pytorch and Jax (using Flax))

`hyper-nn` empowers users with the ability to create easily customizable [Hypernetworks](https://arxiv.org/abs/1609.09106) for almost any generic `nn.Module` from [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and [Flax](https://flax.readthedocs.io/en/latest/flax.linen.html). Our Hypernetwork objects are also `nn.Modules`, allowing for easy integration in existing systems

## What are Hypernetworks?
Hypernetworks, simply put, are neural networks that generate parameters for another neural network. They can be incredibly powerful, being able to represent large networks while using only a fraction of their parameters.

`hyper-nn` represents Hypernetworks with two key components: an Embedding that holds information about layer(s) in the target network and a Weight Generator, which takes in the embedding and outputs a parameter vector for the target network. 

Hypernetworks also come in two variants, static or dynamic. Static Hypernetworks have a fixed or learned embedding and weight generator that outputs the target networks’ weights deterministically. Dynamic Hypernetworks instead receive inputs and use them to generate dynamic weights.

#### Static Hypernetwork

                                                          ┌───────────┐                    
                                                          │   Input   │
                                                          └─────┬─────┘
                                                                │
                                                                ▼
    ┌────────────────────────────────────────────┐  ┌───────────────────────┐
    │                                            │  │                       │
    │              HyperNetwork                  │  │     Target Network    │
    │ ┌───────────┐          ┌─────────────────┐ │  │  ┌─────────────────┐  │
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

#### Dynamic Hypernetwork

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

### Key Components 

#### EmbeddingModule



## Install
```bash
$ pip install hyper-nn
```
For gpu functionality with Jax, you will need to follow the instructions [here](https://github.com/google/jax#installation)

## Usage

For both Pytorch and Jax implementations there are 3 main components:
- EmbeddingModule
- WeightGenerator
- Hypernetwork

[Base Hypernet](hypernn/base_hypernet.py)
```python

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

```

## Pytorch

### Components

##### [EmbeddingModule](hypernn/torch/embedding_module.py)
```python
class TorchEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        ...

    @abc.abstractmethod
    def forward(self, inp: Optional[Any] = None, *args, **kwargs):
        """
        Generate Embedding
        """

```

##### [Weight Generator](hypernn/torch/weight_generator.py)
```python
class TorchWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        ...

    @abc.abstractmethod
    def forward(
        self, embedding: torch.Tensor, inp: Optional[Any] = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Generate Embedding
        """

```

##### [Hypernetwork](build/lib/hypernn/torch/hypernet.py)
```python
class TorchHyperNetwork(nn.Module, BaseHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        embedding_module_constructor: Callable[
            [int, int], TorchEmbeddingModule
        ] = StaticTorchEmbeddingModule,
        weight_generator_constructor: Callable[
            [int, int], TorchWeightGenerator
        ] = LinearTorchWeightGenerator,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        hidden_dim: Optional[int] = None,
    ):

    def forward(
        self, x: Any, params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

```

#### Minimal Example
```python
import torch.nn as nn

# target network
target_network = nn.Sequential(
    nn.Linear(8, 256),
    nn.Tanh(),
    nn.Linear(256,256),
    nn.Tanh(),
    nn.Linear(256, 4, bias=False)
)

from hypernn.torch.hypernet import TorchHyperNetwork
from hypernn.torch.weight_generator import TorchWeightGenerator
from hypernn.torch.embedding_module import TorchEmbeddingModule

# embedding module
class DefaultTorchEmbeddingModule(TorchEmbeddingModule):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__(embedding_dim, num_embeddings)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, *args, **kwargs):
        indices = torch.arange(self.num_embeddings).to(self.device)
        return self.embedding(indices)

# weight generator
class DefaultTorchWeightGenerator(TorchWeightGenerator):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__(embedding_dim, hidden_dim)
        self.generator = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.generator(embedding).view(-1)

# putting it all together

hypernetwork = TorchHyperNetwork(
                            target_network,
                            embedding_module_constructor=DefaultTorchEmbeddingModule,
                            weight_generator_constructor=DefaultTorchWeightGenerator,
                            embedding_dim = 32,
                            num_embeddings = 512
                        )

# now we can use the hypernetwork like any other nn.Module
inp = torch.zeros((1, 8))
output, generated_params = hypernetwork(inp)

# pass in previous generated params
output, _ = hypernetwork(inp, params=generated_params)

```
---

## Jax

For jax implementations, we use the jax neural network library [flax](https://github.com/google/flax)

### Components

##### [EmbeddingModule](hypernn/jax/embedding_module.py)
```python
class FlaxEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    embedding_dim: int
    num_embeddings: int

    @abc.abstractmethod
    def __call__(self, inp: Optional[Any] = None, *args, **kwargs):
        """
        Forward pass to output embeddings
        """

```

##### [Weight Generator](hypernn/jax/weight_generator.py)
```python
class FlaxWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    embedding_dim: int
    hidden_dim: int

    @abc.abstractmethod
    def __call__(
        self, embedding: jnp.array, inp: Optional[Any] = None, *args, **kwargs
    ):
        """
        Forward pass to output embeddings
        """

```

##### [Hypernetwork](hypernn/jax/hypernet.py)
Because flax networks do not automatically initialize weights like torch networks do, we need to pass in input shape into the constructor as an additional parameter

```python
class FlaxHyperNetwork(
    input_shape: Tuple[int, ...],
    target_network_class: nn.Module,
    embedding_module_constructor: Callable[
        [int, int], FlaxEmbeddingModule
    ] = FlaxStaticEmbeddingModule,
    weight_generator_constructor: Callable[
        [int, int], FlaxStaticWeightGenerator
    ] = FlaxStaticWeightGenerator,
    embedding_dim: int = 100,
    num_embeddings: int = 3,
    hidden_dim: Optional[int] = None,
):

        def __call__(
            self, x: Any, params: Optional[List[jnp.array]] = None
        ) -> Tuple[jnp.array, List[jnp.array]]:
            ...
```

#### Example usage
```python
import flax.linen as nn

# target network
class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(4, use_bias=False)(x)
        return x

from hypernn.jax.embedding_module import FlaxEmbeddingModule
from hypernn.jax.weight_generator import FlaxWeightGenerator
from hypernn.jax.hypernet import FlaxHyperNetwork

# embedding module
class DefaultFlaxEmbeddingModule(FlaxEmbeddingModule):
    def setup(self):
        self.embedding = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self):
        indices = jnp.arange(0, self.num_embeddings)
        return self.embedding(indices)

# weight generator
class DefaultFlaxWeightGenerator(FlaxWeightGenerator):
    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim)

    def __call__(self, embedding: jnp.array):
        return self.dense1(embedding)

# putting it all together
hypernetwork = FlaxHyperNetwork(
                            input_shape = (1, 8),
                            target_network = MLP(),
                            embedding_module_constructor=DefaultFlaxEmbeddingModule,
                            weight_generator_constructor=DefaultFlaxWeightGenerator,
                            embedding_dim = 32,
                            num_embeddings = 512
    )

rng = jax.random.PRNGKey(0)
variables = hypernetwork.init(rng, jnp.ones((1,8)))
output, generated_params = hypernetwork.apply(variables, jnp.zeros((1,8)))

# pass in previous generated params
output, _ = hypernetwork.apply(variables, jnp.zeros((1,8)), params=generated_params)

```
---

### Extra

#### Using custom constructors with functools.partial

```python
from functools import partial

class CustomStaticWeightGenerator(TorchWeightGenerator):
    def __init__(self, embedding_dim: int, hidden_dim: int, embedding_network_hidden_dim: int):
        super().__init__(embedding_dim, hidden_dim)
        self.embedding_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_network_hidden_dim),
            nn.ReLU(),
            nn.Linear(embedding_network_hidden_dim, hidden_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.embedding_network(embedding).view(-1)

weight_generator_constructor = partial(CustomStaticWeightGenerator, embedding_network_hidden_dim=64)


hypernetwork = TorchHyperNetwork(
                            target_network,
                            embedding_module_constructor=DefaultTorchEmbeddingModule,
                            weight_generator_constructor=weight_generator_constructor,
                            embedding_dim = 32,
                            num_embeddings = 512
                        )
```
