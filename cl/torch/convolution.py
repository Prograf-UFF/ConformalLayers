from .module import _MinkowskiModuleWrapper, ConformalModule
from .utils import _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from typing import Optional, Tuple
import MinkowskiEngine as me
import numpy, torch


class _WrappedMinkowskiConvolution(me.MinkowskiConvolution):
    def __init__(self, padding: _size_any_t, *args, **kwargs) -> None:
        super(_WrappedMinkowskiConvolution, self).__init__(*args, **kwargs)
        self._padding = torch.as_tensor(padding, dtype=torch.int32)
        self._wrapper = _MinkowskiModuleWrapper(self)

    def _super_forward(self, input: me.SparseTensor, coords: torch.IntTensor) -> me.SparseTensor:
        return super().forward(input, coords)

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        return self._wrapper.forward(input)

    @property
    def padding(self) -> torch.IntTensor:
        return self._padding


class _WrappedMinkowskiConvolutionTranspose(me.MinkowskiConvolutionTranspose):
    def __init__(self, padding: _size_any_t, *args, **kwargs) -> None:
        super(_WrappedMinkowskiConvolutionTranspose, self).__init__(*args, **kwargs)
        self._padding = torch.as_tensor(padding, dtype=torch.int32)
        self._wrapper = _MinkowskiModuleWrapper(self)

    def _super_forward(self, input: me.SparseTensor, coords: torch.IntTensor) -> me.SparseTensor:
        return super().forward(input, coords)

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        return self._wrapper.forward(input)

    @property
    def padding(self) -> torch.IntTensor:
        return self._padding


class ConvNd(ConformalModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 stride: _size_any_t,
                 padding: _size_any_t,
                 dilation: _size_any_t,
                 transposed: bool=False,
                 name: Optional[str]=None) -> None:
        super(ConvNd, self).__init__(name)
        WrappedModuleClass = _WrappedMinkowskiConvolutionTranspose if transposed else _WrappedMinkowskiConvolution
        self._native = WrappedModuleClass(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dimension=len(kernel_size),
            has_bias=False)

    def __repr__(self) -> str:
       return f'Conv(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, dilation={*map(int, self.dilation),}, transposed={self.transposed}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return self.out_channels, tuple(map(int, numpy.floor(numpy.add(numpy.true_divide(numpy.add(in_volume, numpy.subtract(numpy.subtract(numpy.multiply(self.padding, 2), numpy.multiply(self.dilation, numpy.subtract(self.kernel_size, 1))), 1)), self.stride), 1))))

    def _register_parent(self, parent, index: int) -> None:
        parent.register_parameter(f'Conv{index}' if self._name is None else self._name, self._native.kernel)
        self._native.kernel.register_hook(lambda _: parent.invalidate_cache())

    @property
    def in_channels(self) -> int:
        return self._native.in_channels

    @property
    def out_channels(self) -> int:
        return self._native.out_channels

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

    @property
    def transposed(self) -> bool:
        return self._native.is_transpose

    @property
    def kernel(self) -> torch.nn.Parameter:
        return self._native.kernel


class Conv1d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_1_t,
                 stride: _int_or_size_1_t=1,
                 padding: _int_or_size_1_t=0,
                 dilation: _int_or_size_1_t=1,
                 name: Optional[str]=None) -> None:
        super(Conv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            dilation=_single(dilation),
            transposed=False,
            name=name)


class Conv2d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_2_t,
                 stride: _int_or_size_2_t=1,
                 padding: _int_or_size_2_t=0,
                 dilation: _int_or_size_2_t=1,
                 name: Optional[str]=None) -> None:
        super(Conv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            transposed=False,
            name=name)


class Conv3d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_3_t,
                 stride: _int_or_size_3_t=1,
                 padding: _int_or_size_3_t=0,
                 dilation: _int_or_size_3_t=1,
                 name: Optional[str]=None) -> None:
        super(Conv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            dilation=_triple(dilation),
            transposed=False,
            name=name)


class ConvTransposeNd(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 stride: _size_any_t=1,
                 padding: _size_any_t=0,
                 dilation: _size_any_t=1,
                 name: Optional[str]=None) -> None:
        super(ConvTransposeNd, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=True,
            name=name)


class ConvTranspose1d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_1_t,
                 stride: _int_or_size_1_t=1,
                 padding: _int_or_size_1_t=0,
                 dilation: _int_or_size_1_t=1,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            dilation=_single(dilation),
            transposed=True,
            name=name)


class ConvTranspose2d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_2_t,
                 stride: _int_or_size_2_t=1,
                 padding: _int_or_size_2_t=0,
                 dilation: _int_or_size_2_t=1,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            transposed=True,
            name=name)


class ConvTranspose3d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_3_t,
                 stride: _int_or_size_3_t=1,
                 padding: _int_or_size_3_t=0,
                 dilation: _int_or_size_3_t=1,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            dilation=_triple(dilation),
            transposed=True,
            name=name)
