# Hyper-nn (Easy Hypernetworks in Pytorch and Jax (using Flax))

Hyper-nn is a plug-n-play library for [Hypernetworks](https://arxiv.org/abs/1609.09106) in Pytorch and Jax. The implementations are meant to be integrated easily with existing Pytorch and Flax networks.


## Install

```bash
$ pip install hyper-nn
```


## Architecture

#### Static Hypernetwork

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
                                                                ▼
                                                          ┌───────────┐
                                                          │   Output  │
                                                          └───────────┘

#### Dynamic Hypernetwork

                                                          ┌───────────┐
            ┌────────────────────  or ┌───────────────────┤   Input   │
            │                         │                   └─────┬─────┘
            │                         │                         │
            │                         │                         ▼
    ┌───────┼─────────────────────────┼──────────┐  ┌────────────────────────┐
    │       │                         │          │  │                        │
    │       │      HyperNetwork       │          │  │      Target Network    │
    │ ┌─────▼─────┐          ┌────────▼────────┐ │  │  ┌─────────────────┐   │
    │ │           │          │                 │ │  │  │                 │   │
    │ │ Embedding ├─────────►│ Weight Generator├─┼──┼─►│Generated Weights│   │
    │ │           │          │                 │ │  │  │                 │   │
    │ └───────────┘          └─────────────────┘ │  │  └─────────────────┘   │
    │                                            │  │                        │
    └────────────────────────────────────────────┘  └───────────┬────────────┘
                                                                │
                                                                ▼
                                                          ┌───────────┐
                                                          │   Output  │
                                                          └───────────┘

## Usage

#### Pytorch

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
class StaticEmbeddingModule(TorchEmbeddingModule):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__(embedding_dim, num_embeddings)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self):
        indices = torch.arange(self.num_embeddings)
        return self.embedding(indices)

# weight generator
class StaticWeightGenerator(TorchWeightGenerator):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__(embedding_dim, hidden_dim)
        self.embedding_network = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.embedding_network(embedding).view(-1)

# putting it all together

hypernetwork = TorchHyperNetwork(
                            target_network,
                            embedding_module_constructor=StaticEmbeddingModule,
                            weight_generator_constructor=StaticWeightGenerator,
                            embedding_dim = 32,
                            num_embeddings = 512
                        )

# now we can use the hypernetwork like any other nn.Module

# example usage
inp = torch.zeros((1, 8))
output, generated_params = hypernetwork(inp)

```