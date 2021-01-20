from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData, WrappedMinkowskiStridedOperation
from .utils import DenseTensor, ScalarTensor, IntOrSize1, IntOrSize2, IntOrSize3, SizeAny, Pair, Single, Triple
from collections import OrderedDict
from typing import Optional, Tuple, Union
import MinkowskiEngine as me
import torch


class WrappedMinkowskiAvgPooling(WrappedMinkowskiStridedOperation):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedMinkowskiAvgPooling, self).__init__(
            owner,
            transposed=False)
        self._kernel_entry = 1 / float(torch.prod(owner.kernel_size))
        self._function = me.MinkowskiAvgPoolingFunction()
        
    def _apply_function(self, input: me.SparseTensor, alpha_upper: ScalarTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> Tuple[DenseTensor, ScalarTensor]:
        out_feats = self._function.apply(input.feats, input.tensor_stride, 1, self.owner.kernel_size, 1, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)
        out_feats = out_feats * self._kernel_entry
        return out_feats, alpha_upper # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality). Recall that the L1-norm of the mean kernel is 1, so alpha_upper * 1 = alpha_upper.


class AvgPoolNd(ConformalModule):
    _TORCH_MODULE_CLASS = None

    def __init__(self,
                 kernel_size: SizeAny,
                 stride: Optional[SizeAny],
                 padding: SizeAny,
                 *, name: Optional[str]=None) -> None:
        super(AvgPoolNd, self).__init__(name=name)
        self._torch_module = self._TORCH_MODULE_CLASS(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=True)
        self._minkowski_module = WrappedMinkowskiAvgPooling(self)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            return (self._torch_module(input), input_extra), alpha_upper # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality). Recall that the L1-norm of the mean kernel is 1, so alpha_upper * 1 = alpha_upper.
        else:
            return self._minkowski_module(input)

    def output_dims(self, in_channels: int, *in_volume: int) -> SizeAny:
        return (in_channels, *map(int, (torch.as_tensor(in_volume, dtype=torch.int32, device='cpu') + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1))

    @property
    def minkowski_module(self) -> torch.nn.Module:
        return self._minkowski_module

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._torch_module

    @property
    def dilation(self) -> int:
        return 1

    @property
    def kernel_size(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.kernel_size, dtype=torch.int32, device='cpu')

    @property
    def stride(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.stride, dtype=torch.int32, device='cpu')

    @property
    def padding(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.padding, dtype=torch.int32, device='cpu')


class AvgPool1d(AvgPoolNd):
    _TORCH_MODULE_CLASS = torch.nn.AvgPool1d

    def __init__(self,
                 kernel_size: IntOrSize1,
                 stride: Optional[IntOrSize1]=None,
                 padding: IntOrSize1=0,
                 *, name: Optional[str]=None) -> None:
        super(AvgPool1d, self).__init__(
            kernel_size=Single(kernel_size),
            stride=Single(stride),
            padding=Single(padding),
            name=name)


class AvgPool2d(AvgPoolNd):
    _TORCH_MODULE_CLASS = torch.nn.AvgPool2d

    def __init__(self,
                 kernel_size: IntOrSize2,
                 stride: Optional[IntOrSize2]=None,
                 padding: IntOrSize2=0,
                 *, name: Optional[str]=None) -> None:
        super(AvgPool2d, self).__init__(
            kernel_size=Pair(kernel_size),
            stride=Pair(stride),
            padding=Pair(padding),
            name=name)


class AvgPool3d(AvgPoolNd):
    _TORCH_MODULE_CLASS = torch.nn.AvgPool3d

    def __init__(self,
                 kernel_size: IntOrSize3,
                 stride: Optional[IntOrSize3]=None,
                 padding: IntOrSize3=0,
                 *, name: Optional[str]=None) -> None:
        super(AvgPool3d, self).__init__(
            kernel_size=Triple(kernel_size),
            stride=Triple(stride),
            padding=Triple(padding),
            name=name)
