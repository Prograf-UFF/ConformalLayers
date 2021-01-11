from .module import ConformalModule, SimpleMinkowskiModuleWrapper
from collections import OrderedDict
from typing import Optional
import MinkowskiEngine as me
import torch


class Linear(ConformalModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 *, name: Optional[str]=None) -> None:
        super(Linear, self).__init__(
            SimpleMinkowskiModuleWrapper(
                module=me.MinkowskiLinear(in_features=in_features, out_features=out_features, bias=False),
                output_dims=lambda *in_dims: (*in_dims[:-1], out_features)),
            name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['in_features'] = self.in_features
        entries['out_features'] = self.out_features
        return entries

    @property
    def in_features(self) -> int:
        return self.native.module.linear.in_features

    @property
    def out_features(self) -> int:
        return self.native.module.linear.out_features

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.native.module.linear.weight
