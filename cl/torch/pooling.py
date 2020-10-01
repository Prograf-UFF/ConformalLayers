from .module import _MinkowskiModuleWrapper, ConformalModule
from .utils import _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from typing import Optional, Tuple
import MinkowskiEngine as me
import numpy, torch


class _WrappedMinkowskiAvgPooling(me.MinkowskiAvgPooling):
    def __init__(self, padding: _size_any_t, *args, **kwargs) -> None:
        super(_WrappedMinkowskiAvgPooling, self).__init__(*args, **kwargs)
        self._padding = torch.as_tensor(padding, dtype=torch.int32)
        self._wrapper = _MinkowskiModuleWrapper(self)

    def _super_forward(self, input: me.SparseTensor, coords: torch.IntTensor) -> me.SparseTensor:
        return super().forward(input, coords)

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        return self._wrapper.forward(input)

    @property
    def is_transpose(self) -> bool:
        return False
    
    @property
    def padding(self) -> torch.IntTensor:
        return self._padding


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
            dilation=dilation,
            dimension=len(kernel_size))

    def __repr__(self) -> str:
       return f'AvgPool(kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, dilation={*map(int, self.dilation),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, tuple(map(int, numpy.floor(numpy.add(numpy.true_divide(numpy.add(in_volume, numpy.subtract(numpy.subtract(numpy.multiply(self.padding, 2), numpy.multiply(self.dilation, numpy.subtract(self.kernel_size, 1))), 1)), self.stride), 1))))

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

    @property
    def dimension(self) -> int:
        return self._native.dimension


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
                 dilation: _int_or_size_3_t=1,
                 name: Optional[str]=None) -> None:
        super(AvgPool3d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            dilation=_triple(dilation),
            name=name)
