from .module import MinkowskiOperationWrapper, ConformalModule
from .utils import IntOrSize1, IntOrSize2, IntOrSize3, SizeAny, Pair, Single, Triple
from collections import OrderedDict
from typing import Optional, Tuple
import MinkowskiEngine as me
import torch


class WrappedMinkowskiAvgPooling(MinkowskiOperationWrapper):
    def __init__(self, **kwargs) -> None:
        super(WrappedMinkowskiAvgPooling, self).__init__(transposed=False, **kwargs)
        self._inv_cardinality = 1 / int(torch.prod(self.kernel_size))
        self._function = me.MinkowskiAvgPoolingFunction()

    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> torch.Tensor:
        out_feats = self._function.apply(input.feats, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)
        out_feats *= self._inv_cardinality
        return out_feats

    def output_size(self, in_channels: int, in_volume: SizeAny) -> Tuple[int, SizeAny]:
        return in_channels, tuple(map(int, (torch.as_tensor(in_volume, dtype=torch.int32, device='cpu') + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))


class WrappedMinkowskiSumPooling(MinkowskiOperationWrapper):
    def __init__(self, **kwargs) -> None:
        super(WrappedMinkowskiSumPooling, self).__init__(transposed=False, **kwargs)
        self._function = me.MinkowskiAvgPoolingFunction()

    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> torch.Tensor:
        return self._function.apply(input.feats, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)

    def output_size(self, in_channels: int, in_volume: SizeAny) -> Tuple[int, SizeAny]:
        return in_channels, tuple(map(int, (torch.as_tensor(in_volume, dtype=torch.int32, device='cpu') + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))


class AvgPoolNd(ConformalModule):
    def __init__(self,
                 kernel_size: SizeAny,
                 stride: Optional[SizeAny],
                 padding: SizeAny,
                 dilation: SizeAny,
                 *, name: Optional[str]=None) -> None:
        super(AvgPoolNd, self).__init__(
            WrappedMinkowskiAvgPooling(
                kernel_size=kernel_size,
                stride=kernel_size if stride is None else stride,
                padding=padding,
                dilation=dilation),
            name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        entries['dilation'] = tuple(map(int, self.dilation))
        return entries

    @property
    def kernel_size(self) -> torch.IntTensor:
        return self.native.kernel_size

    @property
    def stride(self) -> torch.IntTensor:
        return self.native.stride

    @property
    def padding(self) -> torch.IntTensor:
        return self.native.padding

    @property
    def dilation(self) -> torch.IntTensor:
        return self.native.dilation


class AvgPool1d(AvgPoolNd):
    def __init__(self,
                 kernel_size: IntOrSize1,
                 stride: Optional[IntOrSize1]=None,
                 padding: IntOrSize1=0,
                 dilation: IntOrSize1=1,
                 *, name: Optional[str]=None) -> None:
        super(AvgPool1d, self).__init__(
            kernel_size=Single(kernel_size),
            stride=Single(stride),
            padding=Single(padding),
            dilation=Single(dilation),
            name=name)


class AvgPool2d(AvgPoolNd):
    def __init__(self,
                 kernel_size: IntOrSize2,
                 stride: Optional[IntOrSize2]=None,
                 padding: IntOrSize2=0,
                 dilation: IntOrSize2=1,
                 *, name: Optional[str]=None) -> None:
        super(AvgPool2d, self).__init__(
            kernel_size=Pair(kernel_size),
            stride=Pair(stride),
            padding=Pair(padding),
            dilation=Pair(dilation),
            name=name)


class AvgPool3d(AvgPoolNd):
    def __init__(self,
                 kernel_size: IntOrSize3,
                 stride: Optional[IntOrSize3]=None,
                 padding: IntOrSize3=0,
                 dilation: IntOrSize3=0,
                 *, name: Optional[str]=None) -> None:
        super(AvgPool3d, self).__init__(
            kernel_size=Triple(kernel_size),
            stride=Triple(stride),
            padding=Triple(padding),
            dilation=Triple(dilation),
            name=name)


class SumPoolNd(ConformalModule):
    def __init__(self,
                 kernel_size: SizeAny,
                 stride: Optional[SizeAny],
                 padding: SizeAny,
                 dilation: SizeAny,
                 *, name: Optional[str]=None) -> None:
        super(SumPoolNd, self).__init__(
            WrappedMinkowskiSumPooling(
                kernel_size=kernel_size,
                stride=kernel_size if stride is None else stride,
                padding=padding,
                dilation=dilation),
            name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        entries['dilation'] = tuple(map(int, self.dilation))
        return entries

    @property
    def kernel_size(self) -> torch.IntTensor:
        return self.native.kernel_size

    @property
    def stride(self) -> torch.IntTensor:
        return self.native.stride

    @property
    def padding(self) -> torch.IntTensor:
        return self.native.padding

    @property
    def dilation(self) -> torch.IntTensor:
        return self.native.dilation


class SumPool1d(SumPoolNd):
    def __init__(self,
                 kernel_size: IntOrSize1,
                 stride: Optional[IntOrSize1]=None,
                 padding: IntOrSize1=0,
                 dilation: IntOrSize1=1,
                 *, name: Optional[str]=None) -> None:
        super(SumPool1d, self).__init__(
            kernel_size=Single(kernel_size),
            stride=Single(stride),
            padding=Single(padding),
            dilation=Single(dilation),
            name=name)


class SumPool2d(SumPoolNd):
    def __init__(self,
                 kernel_size: IntOrSize2,
                 stride: Optional[IntOrSize2]=None,
                 padding: IntOrSize2=0,
                 dilation: IntOrSize2=1,
                 *, name: Optional[str]=None) -> None:
        super(SumPool2d, self).__init__(
            kernel_size=Pair(kernel_size),
            stride=Pair(stride),
            padding=Pair(padding),
            dilation=Pair(dilation),
            name=name)


class SumPool3d(SumPoolNd):
    def __init__(self,
                 kernel_size: IntOrSize3,
                 stride: Optional[IntOrSize3]=None,
                 padding: IntOrSize3=0,
                 dilation: IntOrSize3=1,
                 *, name: Optional[str]=None) -> None:
        super(SumPool3d, self).__init__(
            kernel_size=Triple(kernel_size),
            stride=Triple(stride),
            padding=Triple(padding),
            dilation=Triple(dilation),
            name=name)
