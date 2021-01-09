from .module import ConformalModule, NativeModuleWrapper
from .utils import SizeAny
from collections import OrderedDict
from typing import Optional, Tuple
import MinkowskiEngine as me


class WrappedRegularization(NativeModuleWrapper):
    def __init__(self, module: me.MinkowskiModuleBase) -> None:
        super(WrappedRegularization, self).__init__()
        self._module = module

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        return self.module(input)

    def output_size(self, in_channels: int, in_volume: SizeAny) -> Tuple[int, SizeAny]:
        return in_channels, in_volume

    @property
    def module(self) -> me.MinkowskiModuleBase:
        return self._module


class Dropout(ConformalModule):
    def __init__(self,
                 p: float=0.5,
                 inplace: bool=False,
                 *, name: Optional[str]=None) -> None:
        super(Dropout, self).__init__(
            WrappedRegularization(me.MinkowskiDropout(p=p, inplace=inplace)),
            name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['p'] = self.p
        entries['inplace'] = self.inplace
        entries['active'] = self.training
        return entries

    @property
    def p(self) -> float:
        return self.native.module.module.p

    @property
    def inplace(self) -> bool:
        return self.native.module.module.inplace
