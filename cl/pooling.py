from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData, WrappedMinkowskiStridedOperation
from .utils import DenseTensor, ScalarTensor, IntOrSize1, IntOrSize2, IntOrSize3, SizeAny, Pair, Single, Triple
from collections import OrderedDict
from typing import Optional, Tuple, Union
import MinkowskiEngine as me
from MinkowskiEngineBackend._C import PoolingMode
import torch


class WrappedMinkowskiAvgPooling(WrappedMinkowskiStridedOperation):

    def __init__(self, owner: ConformalModule, kernel_generator: me.KernelGenerator) -> None:
        super(WrappedMinkowskiAvgPooling, self).__init__(owner)
        self._function = me.MinkowskiLocalPoolingFunction()
        self._inv_kernel_cardinality = 1 / float(torch.prod(owner.kernel_size))
        self._kernel_generator = kernel_generator
        
    def _apply_function(self, input: me.SparseTensor, alpha_upper: ScalarTensor, out_coordinate_map_key: me.CoordinateMapKey) -> Tuple[DenseTensor, ScalarTensor]:
        out_feats = self._function.apply(input.features, PoolingMode.LOCAL_SUM_POOLING, self._kernel_generator, input.coordinate_map_key, out_coordinate_map_key, input._manager)
        out_feats = out_feats * self._inv_kernel_cardinality
        # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
        # Recall that the L1-norm of the mean kernel is 1, so alpha_upper * 1 = alpha_upper.
        return out_feats, alpha_upper


class AvgPoolNd(ConformalModule):

    _TORCH_MODULE_CLASS = None

    def __init__(self, kernel_size: SizeAny, stride: Optional[SizeAny], padding: SizeAny, *, name: Optional[str] = None) -> None:
        super(AvgPoolNd, self).__init__(name=name)
        self._torch_module = self._TORCH_MODULE_CLASS(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=False, count_include_pad=True)
        self._minkowski_module = WrappedMinkowskiAvgPooling(self, me.KernelGenerator(kernel_size=kernel_size, stride=1, dilation=1, is_transpose=False, expand_coordinates=False, dimension=len(kernel_size)))

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
            # Recall that the L1-norm of the mean kernel is 1, so alpha_upper * 1 = alpha_upper.
            return (self._torch_module(input), input_extra), alpha_upper
        else:
            return self._minkowski_module(input)

    def output_dims(self, in_channels: int, *in_volume: int) -> SizeAny:
        return (in_channels, *map(int, torch.div(torch.as_tensor(in_volume, dtype=torch.int32) + 2 * self.padding - (self.kernel_size - 1) - 1, self.stride, rounding_mode='floor') + 1))

    @property
    def dilation(self) -> int:
        return 1

    @property
    def kernel_size(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.kernel_size, dtype=torch.int32)

    @property
    def stride(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.stride, dtype=torch.int32)

    @property
    def padding(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.padding, dtype=torch.int32)


class AvgPool1d(AvgPoolNd):

    _TORCH_MODULE_CLASS = torch.nn.AvgPool1d

    def __init__(self, kernel_size: IntOrSize1, stride: Optional[IntOrSize1] = None, padding: IntOrSize1 = 0, *, name: Optional[str] = None) -> None:
        super(AvgPool1d, self).__init__(kernel_size=Single(kernel_size), stride=Single(stride), padding=Single(padding), name=name)


class AvgPool2d(AvgPoolNd):

    _TORCH_MODULE_CLASS = torch.nn.AvgPool2d

    def __init__(self, kernel_size: IntOrSize2, stride: Optional[IntOrSize2] = None, padding: IntOrSize2 = 0, *, name: Optional[str] = None) -> None:
        super(AvgPool2d, self).__init__(kernel_size=Pair(kernel_size), stride=Pair(stride), padding=Pair(padding), name=name)


class AvgPool3d(AvgPoolNd):
    
    _TORCH_MODULE_CLASS = torch.nn.AvgPool3d

    def __init__(self, kernel_size: IntOrSize3, stride: Optional[IntOrSize3] = None, padding: IntOrSize3 = 0, *, name: Optional[str] = None) -> None:
        super(AvgPool3d, self).__init__(kernel_size=Triple(kernel_size), stride=Triple(stride), padding=Triple(padding), name=name)
