from .module import ConformalModule, NativeModuleWrapper
from .utils import SizeAny
from collections import OrderedDict
from typing import Optional
import MinkowskiEngine as me
import numpy, torch


class FlattenWrapper(NativeModuleWrapper):
    def __init__(self) -> None:
        super(FlattenWrapper, self).__init__()
        self._start_dim = 1 #TODO Implement start_dim
        self._end_dim = -1 #TODO Implement end_dim

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        return input

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (numpy.prod(in_dims, dtype=int),)

    @property
    def start_dim(self):
        return self._start_dim

    @property
    def end_dim(self):
        return self._end_dim


class Flatten(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(Flatten, self).__init__(FlattenWrapper(), name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['start_dim'] = self.start_dim
        entries['end_dim'] = self.end_dim
        return entries

    @property
    def start_dim(self):
        return self.native.start_dim

    @property
    def end_dim(self):
        return self.native.end_dim
