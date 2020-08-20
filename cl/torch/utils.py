from itertools import repeat
from typing import Tuple, Union
import collections, functools, torch


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


class EyeTensor:
    _instance = None

    def __init__(self) -> None:
        super(EyeTensor, self).__init__()

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

    def __init__(self) -> None:
        super(ZeroTensor, self).__init__()

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


@implements(torch.add)
def _add(lhs: Union[torch.Tensor, EyeTensor, ZeroTensor], rhs: Union[torch.Tensor, EyeTensor, ZeroTensor]) -> Union[torch.Tensor, EyeTensor, ZeroTensor]:
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
def _chain_matmul(*args) -> Union[torch.Tensor, EyeTensor, ZeroTensor]:
    if any(map(lambda arg: isinstance(arg, ZeroTensor), args)):
        return ZeroTensor()
    tensors = (arg for arg in args if not isinstance(arg, EyeTensor))
    if not tensors:
        return EyeTensor()
    else:
        return torch.chain_matmul(*tensors)


@implements(torch.matmul)
def _matmul(lhs: Union[torch.Tensor, EyeTensor, ZeroTensor], rhs: Union[torch.Tensor, EyeTensor, ZeroTensor]) -> Union[torch.Tensor, EyeTensor, ZeroTensor]:
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


def _ntuple(n: int) -> Union[collections.abc.Iterable, Tuple[int, ...]]:
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)

_int_or_size_1_t = Union[int, Tuple[int]]
_int_or_size_2_t = Union[int, Tuple[int, int]]
_int_or_size_3_t = Union[int, Tuple[int, int, int]]
_size_any_t = Tuple[int, ...]
