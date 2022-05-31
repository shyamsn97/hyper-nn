# hyper-nn -- Easy Hypernetworks in Pytorch and Flax
[![PyPi version](https://badgen.net/pypi/v/hyper-nn/)](https://pypi.org/project/hyper-nn/0.1.0/)


**Note: This library is experimental and currently under development - the flax implementations in particular are far from perfect and can be improved. If you have any suggestions on how to improve this library, please open a github issue or feel free to reach out directly!**

`hyper-nn` gives users with the ability to create easily customizable [Hypernetworks](https://arxiv.org/abs/1609.09106) for almost any generic `torch.nn.Module` from [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and `flax.linen.Module` from [Flax](https://flax.readthedocs.io/en/latest/flax.linen.html). Our Hypernetwork objects are also `torch.nn.Modules` and `flax.linen.Modules`, allowing for easy integration with existing systems

<p align="center">Generating Policy Weights for Lunar Lander</p>

<p float="left">
  <img width="54%" src="https://raw.githubusercontent.com/shyamsn97/hyper-nn/main/images/torch_lunar_lander.gif">
  <img width="45%" src="https://raw.githubusercontent.com/shyamsn97/hyper-nn/main/images/LunarLanderWeights.png">
</p>

<br></br>

<p align="center">Dynamic Weights for each character in a name generator</p>

<p float="center" align="center">
  <img width="100%" src="https://raw.githubusercontent.com/shyamsn97/hyper-nn/main/images/DynamicWeights.png">
</p>

---


## Install
`hyper-nn` tested on python 3.8+

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

Hypernetworks, simply put, are neural networks that generate parameters for another neural network. They can be incredibly powerful, being able to represent large networks while using only a fraction of their parameters.

`hyper-nn` represents Hypernetworks with two key components: 
- EmbeddingModule that holds information about layers(s) in the target network, or more generally a chunk of the target networks weights
- Weight Generator, which takes in the embedding and outputs a parameter vector for the target network

Hypernetworks generally come in two variants, static or dynamic. Static Hypernetworks have a fixed or learned embedding and weight generator that outputs the target networks’ weights deterministically. Dynamic Hypernetworks instead receive inputs and use them to generate dynamic weights.

<p align="center">
  <img width="75%" src="https://raw.githubusercontent.com/shyamsn97/hyper-nn/main/images/dynamic_hypernetwork.drawio.svg">
</p>

---
## Quick Usage

for detailed examples see [notebooks](notebooks/)
- [MNIST](notebooks/mnist/)
- [Lunar Lander Reinforce (Vanilla Policy Gradient)](notebooks/reinforce/)
- [Dynamic Hypernetworks for name generation](notebooks/dynamic_hypernetworks/)


The main classes to use are `TorchHyperNetwork` and `JaxHyperNetwork` and those that inherit them. Instead of constructing them directly, use the `from_target` method, shown below. After this you can use the hypernetwork exactly like any other `nn.Module`!

### Pytorch
```python
import torch.nn as nn

# any module
target_network = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)

# static hypernetwork
from hypernn.torch.hypernet import TorchHyperNetwork

EMBEDDING_DIM = 4
NUM_EMBEDDINGS = 32

hypernetwork = TorchHyperNetwork.from_target(
    target_network = target_network,
    embedding_dim = EMBEDDING_DIM,
    num_embeddings = NUM_EMBEDDINGS
)

# now we can use the hypernetwork like any other nn.Module
inp = torch.zeros((1, 32))

# by default we only output what we'd expect from the target network
output = hypernetwork(inp=[inp])

# return aux_output
output, generated_params, aux_output = hypernetwork(inp=[inp], has_aux=True)

# generate params separately
generated_params, aux_output = hypernetwork.generate_params(inp=[inp])
output = hypernetwork(inp=[inp], generated_params=generated_params)
```

### Jax
```python
import flax.linen as nn
import jax.numpy as jnp
from jax import random

# any module
target_network = nn.Sequential(
    [
        nn.Dense(64),
        nn.relu,
        nn.Dense(32)
    ]
)

# static hypernetwork
from hypernn.jax.hypernet import JaxHyperNetwork

EMBEDDING_DIM = 4
NUM_EMBEDDINGS = 32

hypernetwork = JaxHyperNetwork.from_target(
    target_network = target_network,
    embedding_dim = EMBEDDING_DIM,
    num_embeddings = NUM_EMBEDDINGS,
    inputs=jnp.zeros((1, 32)) # jax needs this to initialize target weights
)

# now we can use the hypernetwork like any other nn.Module
inp = jnp.zeros((1, 32)
key = random.PRNGKey(0)
hypernetwork_params = hypernetwork.init(key, inp=[inp)]) # flax needs to initialize hypernetwork parameters first

# by default we only output what we'd expect from the target network
output = hypernetwork.apply(hypernetwork_params, inp=[inp])

# return aux_output
output, generated_params, aux_output = hypernetwork.apply(hypernetwork_params, inp=[inp], has_aux=True)

# generate params separately
generated_params, aux_output = hypernetwork.apply(hypernetwork_params, inp=[inp], method=hypernetwork.generate_params)

output = hypernetwork.apply(inp=[inp], generated_params=generated_params)
```
---

## Detailed Explanation

### EmbeddingModule

The `EmbeddingModule` is used to store information about layers(s) in the target network, or more generally a chunk of the target networks weights. The standard representation is with a matrix of size `num_embeddings x embedding_dim`. `hyper-nn` uses torch's `nn.Embedding` and flax's `nn.Embed` classes to represent this.

### WeightGenerator
`WeightGenerator` takes in the embedding matrix from `EmbeddingModule` and outputs a parameter vector of size `num_target_parameters`, equal to the total number of parameters in the target network. To ensure that the output is equal to `num_target_parameters`, the `WeightGenerator` outputs a matrix of size `num_embeddings x weight_chunk_dim`, where `weight_chunk_dim = num_target_parameters // num_embeddings`, and then flattens it.

### Hypernetwork

the `Hypernetwork` by default uses a `setup` function to initialize the `embedding_module` and `weight_generator` from either user provided modules or the functions: `make_embedding_module`, `make_weight_generator`. This makes it really easy to customize and use your own modules instead of the basic versions provided. `generate_params` is used to generate the target parameters and `forward` combines the generated parameters with the target network to compute a forward pass

Instead of creating the `Hypernetwork` class directly, use `from_target` instead

Base class:
[code](hypernn/base.py)
```python
class HyperNetwork(metaclass=abc.ABCMeta):
    embedding_module = None
    weight_generator = None

    def setup(self) -> None:
        if self.embedding_module is None:
            self.embedding_module = self.make_embedding_module()

        if self.weight_generator is None:
            self.weight_generator = self.make_weight_generator()

    @abc.abstractmethod
    def make_embedding_module(self):
        """
        Makes an embedding module to be used

        Returns:
            a torch.nn.Module or flax.linen.Module that can be used to return an embedding matrix to be used to generate weights
        """

    @abc.abstractmethod
    def make_weight_generator(self):
        """
        Makes an embedding module to be used

        Returns:
            a torch.nn.Module or flax.linen.Module that can be used to return an embedding matrix to be used to generate weights
        """

    @classmethod
    @abc.abstractmethod
    def count_params(
        cls,
        target,
        target_input_shape: Optional[Any] = None,
    ):
        """
        Counts parameters of target nn.Module

        Args:
            target (Union[torch.nn.Module, flax.linen.Module]): _description_
            target_input_shape (Optional[Any], optional): _description_. Defaults to None.
        """

    @classmethod
    @abc.abstractmethod
    def from_target(cls, target, *args, **kwargs) -> HyperNetwork:
        """
        creates hypernetwork from target

        Args:
            cls (_type_): _description_
        """

    @abc.abstractmethod
    def generate_params(self, inp: Optional[Any] = None, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a vector of parameters for target network

        Args:
            inp (Optional[Any], optional): input, may be useful when creating dynamic hypernetworks

        Returns:
            Any: vector of parameters for target network and a dictionary of extra info
        """

    @abc.abstractmethod
    def forward(
        self,
        inp: Iterable[Any] = [],
        generated_params=None,
        has_aux: bool = True,
        *args,
        **kwargs,
    ):
        """
        Computes a forward pass with generated parameters or with parameters that are passed in

        Args:
            inp (Any): input from system
            generated_params (Optional[Union[torch.tensor, jnp.array]], optional): Generated params. Defaults to None.
            has_aux (bool): flag to indicate whether to return auxiliary info
        Returns:
            returns output and generated params and auxiliary info if has_aux is provided
        """
```

---
### Citation

If you use this software in your academic work please cite

``` 
@misc{sudhakaran2022,
  author = {Sudhakaran, Shyam Sudhakaran},
  title = {hyper-nn},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shyamsn97/hyper-nn}}
}
```