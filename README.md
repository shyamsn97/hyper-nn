# hyper-nn (Easy Hypernetworks in Pytorch and Jax)
**Note: This library is experimental and currently under development - the jax implementations in particular are far from perfect and can be improved. If you have any suggestions on how to improve this library, please open a github issue or feel free to reach out directly!**

`hyper-nn` gives users with the ability to create easily customizable [Hypernetworks](https://arxiv.org/abs/1609.09106) for almost any generic `torch.nn.Module` from [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and `flax.linen.Module` from [Flax](https://flax.readthedocs.io/en/latest/flax.linen.html). Our Hypernetwork objects are also `torch.nn.Modules` and `flax.linen.Modules`, allowing for easy integration with existing systems


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


## Install
`hyper-nn` works with python 3.8+

#### Installing with pip
```bash
$ pip install hyper-nn
```

#### Installing from source
```bash
$ git clone git@github.com:shyamsn97/hyper-nn.git
$ cd hyper-nn
$ python setup.py install
```

For gpu functionality with Jax, you will need to follow the instructions [here](https://github.com/google/jax#installation)

### Key Components 

#### EmbeddingModule
`EmbeddingModule` outputs a matrix of size `num_embeddings x embedding_dim`, where each row contains some information about layer(s) in the target network.

<details><summary> <b>Pytorch</b> </summary>
<p>

[code](hypernn/torch/embedding_module.py)

```python
class TorchEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = None
        self.__device_param_dummy__ = nn.Parameter(torch.empty(0)) # to keep track of device

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    @abc.abstractmethod
    def forward(self, inp: Optional[Any] = None, *args, **kwargs) -> torch.Tensor:
        """
        Generate Embedding
        """

```

</p>
</details>

<details><summary> <b>Jax</b> </summary>
<p>

[code](hypernn/jax/embedding_module.py)

```python
class FlaxEmbeddingModule(nn.Module, metaclass=abc.ABCMeta):
    embedding_dim: int
    num_embeddings: int

    def setup(self):
        pass

    @abc.abstractmethod
    def __call__(self, inp: Optional[Any] = None, *args, **kwargs) -> jnp.array:
        """
        Forward pass to output embeddings
        """

```

</p>
</details>

#### WeightGenerator
`WeightGenerator` takes in the embedding matrix from `EmbeddingModule` and outputs a parameter vector of size `num_target_parameters`, equal to the total number of parameters in the target network. To ensure that the output is equal to `num_target_parameters`, the `WeightGenerator` outputs a matrix of size `num_embeddings x hidden_dim`, where `hidden_dim = num_target_parameters // num_embeddings`, and then flattens it.

<details><summary> <b>Pytorch</b> </summary>
<p>

[code](hypernn/torch/weight_generator.py)

```python
class TorchWeightGenerator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.__device_param_dummy__ = nn.Parameter(torch.empty(0)) # to keep track of device

    @abc.abstractmethod
    def forward(
        self, embedding: torch.Tensor, inp: Optional[Any] = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Generate Embedding
        """

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

```

</p>
</details>

<details><summary> <b>Jax</b> </summary>
<p>

[code](hypernn/jax/weight_generator.py)

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

</p>
</details>

#### HyperNetwork
`HyperNetwork` takes in a target network, represented as a `torch.nn.Module or flax.linen.Module`, an `EmbeddingModule` and `WeightGenerator` constructor, `embedding_dim`,  and `num_embeddings`. `HyperNetwork` creates the `EmbeddingModule` and `WeightGenerator`, and uses them to generate parameters to be passed into the target network.

[code](hypernn/base_hypernet.py)

```python
class BaseHyperNetwork(metaclass=abc.ABCMeta):
    def __init__(
        self,
        target_network: Union[torch.nn.Module, flax.linen.Module],
        embedding_module_constructor: Callable[
            [int, int], Union[TorchEmbeddingModule, FlaxEmbeddingModule]
        ],
        weight_generator_constructor: Callable[
            [int, int], Union[TorchWeightGenerator, FlaxWeightGenerator]
        ],
        embedding_dim: int,
        num_embeddings: int,
    ):
        pass

    @abc.abstractmethod
    def generate_params(self, inp: Optional[Any] = None, *args, **kwargs) -> Any:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network
        """

    @abc.abstractmethod
    def forward(
        self,
        inp: Any,
        params: Optional[Union[torch.tensor, jnp.array]] = None,
        **kwargs
    ):
        """
        Computes a forward pass with generated parameters or with parameters that are passed in

        Args:
            inp (Any): input from system
            params (Optional[Union[torch.tensor, jnp.array]], optional): Generated params. Defaults to None.
        Returns:
            returns output and generated parameters
        """

```


## Usage
### Minimal Examples

<details><summary> <b>Pytorch</b> </summary>
<p>

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
</p>
</details>

<details><summary> <b>Jax</b> </summary>
<p>

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
</p>
</details>



## Extra

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
