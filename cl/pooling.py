from .module import _MinkowskiOperationWrapper, ConformalModule
from .utils import _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from typing import Optional, Tuple
import MinkowskiEngine as me
import torch


class _WrappedMinkowskiAvgPooling(_MinkowskiOperationWrapper):
    def __init__(self, **kwargs) -> None:
        super(_WrappedMinkowskiAvgPooling, self).__init__(transposed=False, **kwargs)
        self._inv_cardinality = 1 / int(torch.prod(self.kernel_size))
        self._function = me.MinkowskiAvgPoolingFunction()

    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> torch.Tensor:
        out_feats = self._function.apply(input.feats, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)
        out_feats *= self._inv_cardinality
        return out_feats


class _WrappedMinkowskiSumPooling(_MinkowskiOperationWrapper):
    def __init__(self, **kwargs) -> None:
        super(_WrappedMinkowskiSumPooling, self).__init__(transposed=False, **kwargs)
        self._function = me.MinkowskiAvgPoolingFunction()

    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> torch.Tensor:
        return self._function.apply(input.feats, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)


class AvgPoolNd(ConformalModule):
    def __init__(self,
                 kernel_size: _size_any_t,
                 stride: Optional[_size_any_t],
                 padding: _size_any_t,
                 dilation: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(AvgPoolNd, self).__init__(name)
        self._native = _WrappedMinkowskiAvgPooling(
            kernel_size=kernel_size,
            stride=kernel_size if stride is None else stride,
            padding=padding,
            dilation=dilation)

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, dilation={*map(int, self.dilation),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, tuple(map(int, (torch.as_tensor(in_volume) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))

    @property
    def kernel_size(self) -> torch.IntTensor:
        return self._native.kernel_size

    @property
    def stride(self) -> torch.IntTensor:
        return self._native.stride

    @property
    def padding(self) -> torch.IntTensor:
        return self._native.padding

    @property
    def dilation(self) -> torch.IntTensor:
        return self._native.dilation


class AvgPool1d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_1_t,
                 stride: Optional[_int_or_size_1_t]=None,
                 padding: _int_or_size_1_t=0,
                 dilation: _int_or_size_1_t=1,
                 name: Optional[str]=None) -> None:
        super(AvgPool1d, self).__init__(
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            dilation=_single(dilation),
            name=name)


class AvgPool2d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_2_t,
                 stride: Optional[_int_or_size_2_t]=None,
                 padding: _int_or_size_2_t=0,
                 dilation: _int_or_size_2_t=1,
                 name: Optional[str]=None) -> None:
        super(AvgPool2d, self).__init__(
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            name=name)


class AvgPool3d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_3_t,
                 stride: Optional[_int_or_size_3_t]=None,
                 padding: _int_or_size_3_t=0,
                 dilation: _int_or_size_3_t=0,
                 name: Optional[str]=None) -> None:
        super(AvgPool3d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            dilation=_triple(dilation),
            name=name)


class SumPoolNd(ConformalModule):
    def __init__(self,
                 kernel_size: _size_any_t,
                 stride: Optional[_size_any_t],
                 padding: _size_any_t,
                 dilation: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(SumPoolNd, self).__init__(name)
        self._native = _WrappedMinkowskiSumPooling(
            kernel_size=kernel_size,
            stride=kernel_size if stride is None else stride,
            padding=padding,
            dilation=dilation)

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, dilation={*map(int, self.dilation),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, tuple(map(int, (torch.as_tensor(in_volume) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))

    @property
    def kernel_size(self) -> torch.IntTensor:
        return self._native.kernel_size

    @property
    def stride(self) -> torch.IntTensor:
        return self._native.stride

    @property
    def padding(self) -> torch.IntTensor:
        return self._native.padding

    @property
    def dilation(self) -> torch.IntTensor:
        return self._native.dilation


class SumPool1d(SumPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_1_t,
                 stride: Optional[_int_or_size_1_t]=None,
                 padding: _int_or_size_1_t=0,
                 dilation: _int_or_size_1_t=1,
                 name: Optional[str]=None) -> None:
        super(SumPool1d, self).__init__(
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            dilation=_single(dilation),
            name=name)


class SumPool2d(SumPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_2_t,
                 stride: Optional[_int_or_size_2_t]=None,
                 padding: _int_or_size_2_t=0,
                 dilation: _int_or_size_2_t=1,
                 name: Optional[str]=None) -> None:
        super(SumPool2d, self).__init__(
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            name=name)


class SumPool3d(SumPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_3_t,
                 stride: Optional[_int_or_size_3_t]=None,
                 padding: _int_or_size_3_t=0,
                 dilation: _int_or_size_3_t=1,
                 name: Optional[str]=None) -> None:
        super(SumPool3d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            dilation=_triple(dilation),
            name=name)
