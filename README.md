# hyper-nn -- Easy Hypernetworks in Pytorch and Flax
[![PyPi version](https://badgen.net/pypi/v/hyper-nn/)](https://pypi.org/project/hyper-nn/0.1.0/)


**Note: This library is experimental and currently under development - the flax implementations in particular are far from perfect and can be improved. If you have any suggestions on how to improve this library, please open a github issue or feel free to reach out directly!**

`hyper-nn` gives users with the ability to create easily customizable [Hypernetworks](https://arxiv.org/abs/1609.09106) for almost any generic `torch.nn.Module` from [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and `flax.linen.Module` from [Flax](https://flax.readthedocs.io/en/latest/flax.linen.html). Our Hypernetwork objects are also `torch.nn.Modules` and `flax.linen.Modules`, allowing for easy integration with existing systems. For Pytorch, we make use of the amazing library [`functorch`](https://github.com/pytorch/functorch)

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

---
## What are Hypernetworks?
[Hypernetworks](https://arxiv.org/abs/1609.09106), simply put, are neural networks that generate parameters for another neural network. They can be incredibly powerful, being able to represent large networks while using only a fraction of their parameters.

Hypernetworks generally come in two variants, static or dynamic. Static Hypernetworks have a fixed or learned embedding and weight generator that outputs the target networks’ weights deterministically. Dynamic Hypernetworks instead receive inputs and use them to generate dynamic weights.

<p align="center">
  <img width="75%" src="https://raw.githubusercontent.com/shyamsn97/hyper-nn/main/images/dynamic_hypernetwork.drawio.svg">
</p>


`hyper-nn` represents Hypernetworks with two key components: 
- `embedding_module` that holds information about layers(s) in the target network, or more generally a chunk of the target networks weights
- `weight_generator`, which takes in the embedding and outputs a parameter vector for the target network


Both `embedding_module` and `weight_generator` are represented as `torch.nn.Module` and `flax.linen.Module` objects. a `Module` can be passed in as `custom_embedding_module` or `custom_weight_generator`, or it can be defined in the methods `make_embedding_module` or `make_weight_generator`. 

The `generate_params` method feeds the output from `embedding_module` into `weight_generator` to output the target parameters.

The `forward` method takes in a list of inputs and uses the generated parameters to calculate the output. This method acts as the main method for both jax and torch hypernetworks

### [Torch Hypernetwork](hypernn/torch/hypernet.py#L81)
```python
...
  def make_embedding_module(self) -> nn.Module:
      return nn.Embedding(self.num_embeddings, self.embedding_dim)

  def make_weight_generator(self) -> nn.Module:
      return nn.Linear(self.embedding_dim, self.weight_chunk_dim)

  def generate_params(
      self, inp: Iterable[Any] = []
  ) -> Tuple[jnp.array, Dict[str, Any]]:
      embedding = self.embedding_module(jnp.arange(0, self.num_embeddings))
      generated_params = self.weight_generator(embedding).reshape(-1)
      return generated_params, {"embedding": embedding}

  def forward(
      self,
      inp: Iterable[Any] = [],
      generated_params: Optional[torch.Tensor] = None,
      has_aux: bool = False,
      assert_parameter_shapes: bool = True,
      *args,
      **kwargs,
  ):
      """
      Main method for creating / using generated parameters and passing in input into the target network

      Args:
          inp (Iterable[Any], optional): List of inputs to be passing into the target network. Defaults to [].
          generated_params (Optional[torch.Tensor], optional): Generated parameters of the target network. If not provided, the hypernetwork will generate the parameters. Defaults to None.
          has_aux (bool, optional): If True, return the auxiliary output from generate_params method. Defaults to False.
          assert_parameter_shapes (bool, optional): If True, raise an error if generated_params does not have shape (num_target_parameters,). Defaults to True.

      Returns:
          output (torch.Tensor) | (torch.Tensor, Dict[str, torch.Tensor]): returns output from target network and optionally auxiliary output.
      """
      aux_output = {}
      if generated_params is None:
          generated_params, aux_output = self.generate_params(inp, *args, **kwargs)

      if assert_parameter_shapes:
          self.assert_parameter_shapes(generated_params)

      if has_aux:
          return self._target(generated_params, *inp), generated_params, aux_output
      return self._target(generated_params, *inp)

...
```
### [Flax Hypernetwork](hypernn/jax/hypernet.py#L76)
```python
...
  def make_embedding_module(self):
      return nn.Embed(
          self.num_embeddings,
          self.embedding_dim,
          embedding_init=jax.nn.initializers.uniform(),
      )

  def make_weight_generator(self):
      return nn.Dense(self.weight_chunk_dim)

  def generate_params(
      self, inp: Iterable[Any] = []
  ) -> Tuple[torch.Tensor, Dict[str, Any]]:
      embedding = self.embedding_module(
          torch.arange(self.num_embeddings, device=self.device)
      )
      generated_params = self.weight_generator(embedding).view(-1)
      return generated_params, {"embedding": embedding}

  def forward(
      self,
      inp: Iterable[Any] = [],
      generated_params: Optional[jnp.array] = None,
      has_aux: bool = False,
      assert_parameter_shapes: bool = True,
      *args,
      **kwargs,
  ) -> Tuple[jnp.array, List[jnp.array]]:
      """
      Main method for creating / using generated parameters and passing in input into the target network

      Args:
          inp (Iterable[Any], optional): List of inputs to be passing into the target network. Defaults to [].
          generated_params (Optional[jnp.array], optional): Generated parameters of the target network. If not provided, the hypernetwork will generate the parameters. Defaults to None.
          has_aux (bool, optional): If True, return the auxiliary output from generate_params method. Defaults to False.
          assert_parameter_shapes (bool, optional): If True, raise an error if generated_params does not have shape (num_target_parameters,). Defaults to True.

      Returns:
          output (torch.Tensor) | (jnp.array, Dict[str, jnp.array]): returns output from target network and optionally auxiliary output.
      """
      aux_output = {}
      if generated_params is None:
          generated_params, aux_output = self.generate_params(inp, *args, **kwargs)

      if assert_parameter_shapes:
          self.assert_parameter_shapes(generated_params)

      param_tree = create_param_tree(
          generated_params, self.target_weight_shapes, self.target_treedef
      )

      if has_aux:
          return (
              target_forward(self.target_network.apply, param_tree, *inp),
              generated_params,
              aux_output,
          )
      return target_forward(self.target_network.apply, param_tree, *inp)

...
```

---
## Quick Usage

for detailed examples see [notebooks](notebooks/)
- [Generating weights for a CNN on MNIST](notebooks/mnist/)
- [Lunar Lander Reinforce (Vanilla Policy Gradient)](notebooks/reinforce/)
- [Dynamic Hypernetworks for name generation](notebooks/dynamic_hypernetworks/)


The main classes to use are `TorchHyperNetwork` and `JaxHyperNetwork` and those that inherit them. Instead of constructing them directly, use the `from_target` method, shown below. After this you can use the hypernetwork exactly like any other `nn.Module`!

`hyper-nn` also makes it easy to create Dynamic Hypernetworks that use inputs to create target weights. Basic implementations (both < 100 lines) are provided with `JaxDynamicHyperNetwork` and `TorchDynamicHyperNetwork`, which use an rnn and current input to generate weights.

### Pytorch
```python
import torch.nn as nn

from hypernn.torch.hypernet import TorchHyperNetwork
from hypernn.torch.dynamic_hypernet import TorchDynamicHyperNetwork

# any module
target_network = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)

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


### Dynamic Hypernetwork

dynamic_hypernetwork = TorchDynamicHyperNetwork.from_target(
    input_dim = 32,
    target_network = target_network,
    embedding_dim = EMBEDDING_DIM,
    num_embeddings = NUM_EMBEDDINGS
)

# by default we only output what we'd expect from the target network
output = dynamic_hypernetwork(inp=[inp])

```

### Jax
```python
import flax.linen as nn
import jax.numpy as jnp
from jax import random

from hypernn.jax.dynamic_hypernet import JaxHyperNetwork
from hypernn.jax.dynamic_hypernet import JaxDynamicHyperNetwork

# any module
target_network = nn.Sequential(
    [
        nn.Dense(64),
        nn.relu,
        nn.Dense(32)
    ]
)

EMBEDDING_DIM = 4
NUM_EMBEDDINGS = 32

hypernetwork = JaxHyperNetwork.from_target(
    target_network = target_network,
    embedding_dim = EMBEDDING_DIM,
    num_embeddings = NUM_EMBEDDINGS,
    inputs=jnp.zeros((1, 32)) # jax needs this to initialize target weights
)

# now we can use the hypernetwork like any other nn.Module
inp = jnp.zeros((1, 32))
key = random.PRNGKey(0)
hypernetwork_params = hypernetwork.init(key, inp=[inp]) # flax needs to initialize hypernetwork parameters first

# by default we only output what we'd expect from the target network
output = hypernetwork.apply(hypernetwork_params, inp=[inp])

# return aux_output
output, generated_params, aux_output = hypernetwork.apply(hypernetwork_params, inp=[inp], has_aux=True)

# generate params separately
generated_params, aux_output = hypernetwork.apply(hypernetwork_params, inp=[inp], method=hypernetwork.generate_params)

output = hypernetwork.apply(hypernetwork_params, inp=[inp], generated_params=generated_params)


### Dynamic Hypernetwork

dynamic_hypernetwork = JaxDynamicHyperNetwork.from_target(
    input_dim = 32,
    target_network = target_network,
    embedding_dim = EMBEDDING_DIM,
    num_embeddings = NUM_EMBEDDINGS,
    inputs=jnp.zeros((1, 32)) # jax needs this to initialize target weights
)
dynamic_hypernetwork_params = dynamic_hypernetwork.init(key, inp=[inp]) # flax needs to initialize hypernetwork parameters first

# by default we only output what we'd expect from the target network
output = dynamic_hypernetwork.apply(dynamic_hypernetwork_params, inp=[inp])

```

## Customizing Hypernetworks
The main components to modify are the `embedding_module` and the `weight_generator`, controlled by `make_embedding_module`, `make_weight_generator` and `generate_params`. This allows for complete control over how the hypernetwork generates parameters

For example, here we have a hypernetwork that adds a user inputted noise vector to the generated params

```python
from typing import Optional, Iterable
import torch
import torch.nn as nn
# static hypernetwork
from hypernn.torch.hypernet import TorchHyperNetwork

class CustomHypernetwork(TorchHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        num_target_parameters: Optional[int] = None,
        embedding_dim: int = 100,
        num_embeddings: int = 3,
        weight_chunk_dim: Optional[int] = None,
    ):
        super().__init__(
                    target_network = target_network,
                    num_target_parameters = num_target_parameters,
                    embedding_dim = embedding_dim,
                    num_embeddings = num_embeddings,
                    weight_chunk_dim = weight_chunk_dim,
                )

    def generate_params(
        self, inp: Iterable[Any] = [], noise: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        embedding = self.embedding_module(
            torch.arange(self.num_embeddings, device=self.device)
        )
        generated_params = self.weight_generator(embedding).view(-1) + noise
        return generated_params, {"embedding": embedding}


# usage
target_network = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)

EMBEDDING_DIM = 4
NUM_EMBEDDINGS = 32

hypernetwork = TorchHyperNetwork.from_target(
    target_network = target_network,
    embedding_dim = EMBEDDING_DIM,
    num_embeddings = NUM_EMBEDDINGS
)
inp = torch.zeros((1, 32))

noise = torch.randn((hypernetwork.num_target_parameters,))

out = hypernetwork(inp=[inp], noise=noise)
```

---
## Advanced: Using vmap for batching operations
This is useful when dealing with dynamic hypernetworks that generate different params depending on inputs.

### Pytorch
```python
import torch.nn as nn
from functorch import vmap

# dynamic hypernetwork
from hypernn.torch.dynamic_hypernet import TorchDynamicHyperNetwork

# any module
target_network = nn.Sequential(
    nn.Linear(8, 256),
    nn.ReLU(),
    nn.Linear(256, 32)
)

EMBEDDING_DIM = 4
NUM_EMBEDDINGS = 32

# conditioned on input to generate param vector
hypernetwork = TorchDynamicHyperNetwork.from_target(
    target_network = target_network,
    embedding_dim = EMBEDDING_DIM,
    num_embeddings = NUM_EMBEDDINGS,
    input_dim = 8
)

# batch of 10 inputs
inp = torch.randn((10, 1, 8))

# use with a for loop
outputs = []
for i in range(10):
    outputs.append(hypernetwork(inp=[inp[i]]))
outputs = torch.stack(outputs)
assert outputs.size() == (10, 1, 32)

# using vmap
outputs = vmap(hypernetwork)([inp])
assert outputs.size() == (10, 1, 32)
```

## Detailed Explanation

### embedding_module

The `embedding_module` is used to store information about layers(s) in the target network, or more generally a chunk of the target networks weights. The standard representation is with a matrix of size `num_embeddings x embedding_dim`. `hyper-nn` uses torch's `nn.Embedding` and flax's `nn.Embed` classes to represent this.

### weight_generator
`weight_generator` takes in the embedding matrix from `embedding_module` and outputs a parameter vector of size `num_target_parameters`, equal to the total number of parameters in the target network. To ensure that the output is equal to `num_target_parameters`, the `weight_generator` outputs a matrix of size `num_embeddings x weight_chunk_dim`, where `weight_chunk_dim = num_target_parameters // num_embeddings`, and then flattens it.

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
## Citing hyper-nn

If you use this software in your academic work please cite

```bibtex
@misc{sudhakaran2022,
  author = {Sudhakaran, Shyam Sudhakaran},
  title = {hyper-nn},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shyamsn97/hyper-nn}}
}
```
---

### Projects used in hyper-nn
```bibtex
@Misc{functorch2021,
  author =       {Horace He, Richard Zou},
  title =        {functorch: JAX-like composable function transforms for PyTorch},
  howpublished = {\url{https://github.com/pytorch/functorch}},
  year =         {2021}
}
```