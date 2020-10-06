from itertools import repeat
from typing import Iterable, Tuple, Union


def _ntuple(n: int) -> Union[Iterable[int], Tuple[int, ...]]:
    def parse(arg):
        if isinstance(arg, Iterable):
            return arg
        return tuple(repeat(arg, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)

_int_or_size_1_t = Union[int, Tuple[int]]
_int_or_size_2_t = Union[int, Tuple[int, int]]
_int_or_size_3_t = Union[int, Tuple[int, int, int]]
_size_any_t = Tuple[int, ...]
