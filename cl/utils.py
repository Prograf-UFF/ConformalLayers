from itertools import repeat
from typing import Iterable, Tuple, Union
import numpy, torch


def NTuple(n: int) -> Union[Iterable[int], Tuple[int, ...]]:
    def parse(arg):
        if isinstance(arg, Iterable):
            return arg
        return tuple(repeat(arg, n))
    return parse


Single = NTuple(1)
Pair = NTuple(2)
Triple = NTuple(3)

IntOrSize1 = Union[int, Tuple[int]]
IntOrSize2 = Union[int, Tuple[int, int]]
IntOrSize3 = Union[int, Tuple[int, int, int]]
SizeAny = Tuple[int, ...]


def ravel_multi_index(multi_index: Tuple[torch.Tensor, ...], dims: Tuple[int, ...]) -> torch.Tensor:
    out = multi_index[-1].clone().detach()
    for ind, stride in zip(multi_index[-2::-1], numpy.cumprod(dims[:0:-1])):
        out += ind * stride
    return out


def unravel_index(index: torch.Tensor, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
