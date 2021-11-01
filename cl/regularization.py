from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData
from .utils import SizeAny
from collections import OrderedDict
from typing import Optional, Union
import MinkowskiEngine as me


class Dropout(ConformalModule):
    
    def __init__(self,
                 p: float = 0.5,
                 inplace: bool = False,
                 *, name: Optional[str] = None) -> None:
        super(Dropout, self).__init__(name=name)
        self._minkowski_module = me.MinkowskiDropout(p=p, inplace=inplace)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['p'] = self.p
        entries['inplace'] = self.inplace
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData])\
            -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            return (self._minkowski_module.module(input), input_extra), alpha_upper
        else:
            input, alpha_upper = input
            return self._minkowski_module(input), alpha_upper

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (*in_dims,)

    @property
    def p(self) -> float:
        return self._minkowski_module.module.p

    @property
    def inplace(self) -> bool:
        return self._minkowski_module.module.inplace
