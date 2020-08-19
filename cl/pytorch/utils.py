from typing import Union
import functools, torch


HANDLED_FUNCTIONS = {}


class EyeTensor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "EyeTensor()"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, EyeTensor, ZeroTensor)) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def clone(self):
        return self


class ZeroTensor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
       return "ZeroTensor()"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, EyeTensor, ZeroTensor)) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def clone(self):
        return self


def implements(torch_function):
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


@implements(torch.add)
def add(lhs: Union[torch.Tensor, EyeTensor, ZeroTensor], rhs: Union[torch.Tensor, EyeTensor, ZeroTensor]):
    if isinstance(lhs, ZeroTensor):
        if isinstance(rhs, ZeroTensor):
            return ZeroTensor()
        else:
            return rhs.clone()
    else:
        if isinstance(rhs, ZeroTensor):
            return lhs.clone()
        else:
            return torch.add(lhs, rhs)  # Not implemented cases: add(torch.Tensor, EyeTensor), add(EyeTensor, torch.Tensor), add(EyeTensor, EyeTensor)


@implements(torch.chain_matmul)
def chain_matmul(*args):
    if any(map(lambda arg: isinstance(arg, ZeroTensor), args)):
        return ZeroTensor()
    tensors = (arg for arg in args if not isinstance(arg, EyeTensor))
    if not tensors:
        return EyeTensor()
    else:
        return torch.chain_matmul(*tensors)


@implements(torch.matmul)
def matmul(lhs: Union[torch.Tensor, EyeTensor, ZeroTensor], rhs: Union[torch.Tensor, EyeTensor, ZeroTensor]):
    if isinstance(lhs, ZeroTensor) or isinstance(rhs, ZeroTensor):
        return ZeroTensor()
    if isinstance(lhs, EyeTensor):
        if isinstance(rhs, EyeTensor):
            return EyeTensor()
        else:
            return rhs.clone()
    else:
        if isinstance(rhs, EyeTensor):
            return lhs.clone()
        else:
            return torch.matmul(lhs, rhs)
