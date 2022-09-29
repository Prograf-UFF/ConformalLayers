from audioop import bias
from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData
from .utils import SizeAny, ravel_multi_index, unravel_index
from collections import OrderedDict
from typing import Optional, Union
import MinkowskiEngine as me
import numpy
import torch


class Identity(ConformalModule):
    
    def __init__(self, *args, name: Optional[str] = None, **kwargs) -> None:
        super(Identity, self).__init__(name=name)

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        return input

    def output_dims(self, *in_dims: int) -> SizeAny:
        return in_dims
