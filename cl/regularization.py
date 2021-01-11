from .module import ConformalModule, SimpleMinkowskiModuleWrapper
from collections import OrderedDict
from typing import Optional
import MinkowskiEngine as me


class Dropout(ConformalModule):
    def __init__(self,
                 p: float=0.5,
                 inplace: bool=False,
                 *, name: Optional[str]=None) -> None:
        super(Dropout, self).__init__(
            SimpleMinkowskiModuleWrapper(
                module=me.MinkowskiDropout(p=p, inplace=inplace),
                output_dims=lambda *in_dims: (*in_dims,)),
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
