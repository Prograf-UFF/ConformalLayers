from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData
from .utils import SizeAny
from collections import OrderedDict
from typing import Optional, Union
import MinkowskiEngine as me
import torch


class Linear(ConformalModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 *, name: Optional[str]=None) -> None:
        super(Linear, self).__init__(name=name)
        self._minkowski_module = me.MinkowskiLinear(in_features=in_features, out_features=out_features, bias=False)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['in_features'] = self.in_features
        entries['out_features'] = self.out_features
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            output = self._minkowski_module.linear(input)
            alpha_upper = alpha_upper * torch.linalg.norm(self._minkowski_module.linear.weight, ord='fro') # Apply the submultiplicative property of matrix norms, and the property that relates the L2-norm and the Frobenius norm of matrices (https://www.math.usm.edu/lambers/mat610/sum10/lecture2.pdf).
            return (output, input_extra), alpha_upper
        else:
            input, alpha_upper = input
            alpha_upper = alpha_upper * torch.linalg.norm(self._minkowski_module.linear.weight, ord='fro') # Apply the submultiplicative property of matrix norms, and the property that relates the L2-norm and the Frobenius norm of matrices (https://www.math.usm.edu/lambers/mat610/sum10/lecture2.pdf).
            return (output, input_extra), alpha_upper

    def output_dims(self, *in_size: int) -> SizeAny:
        return (*in_size[:-1], self.out_features)

    @property
    def minkowski_module(self) -> torch.nn.Module:
        return self._minkowski_module

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._minkowski_module.linear

    @property
    def in_features(self) -> int:
        return self._minkowski_module.linear.in_features

    @property
    def out_features(self) -> int:
        return self._minkowski_module.linear.out_features

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._minkowski_module.linear.weight
