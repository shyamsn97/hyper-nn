import numpy as np
import torch
import torch.nn as nn


# from https://www.sscardapane.it/tutorials/functorch/
# TODO use functorch once its completely working, still getting error: undefined symbol: _ZNK3c104Type14isSubtypeOfExtERKS0_PSo
def count_params(module: nn.Module):
    return sum([np.prod(p.size()) for p in module.parameters()])


def del_attr(obj, names) -> None:
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val) -> None:
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def module_to_functional(module: nn.Module):

    # Remove all the parameters in the model
    names = []
    named_params = list(module.named_parameters())
    for name, p in list(module.named_parameters()):
        del_attr(module, name.split("."))
        names.append(name)

    def functional_module_fw(params, x):
        for name, p in zip(names, params):
            set_attr(module, name.split("."), p)
        return module(x)

    return functional_module_fw, named_params


class FunctionalParamVectorWrapper(nn.Module):
    """
    This wraps a module so that it takes params in the forward pass
    """

    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.functional, self.named_params = module_to_functional(module)

    def forward(self, param_vector: torch.Tensor, x: torch.Tensor):
        params = []
        start = 0
        for name, p in self.named_params:
            end = start + np.prod(p.size())
            params.append(param_vector[start:end].view(p.size()))
            start = end
        return self.functional(params, x)
