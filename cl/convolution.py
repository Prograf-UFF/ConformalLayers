from .module import _MinkowskiOperationWrapper, ConformalModule
from .utils import _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from typing import Optional, Tuple
import MinkowskiEngine as me
import torch


class _WrappedMinkowskiConvolution(_MinkowskiOperationWrapper):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(_WrappedMinkowskiConvolution, self).__init__(transposed=False, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = torch.nn.Parameter(torch.FloatTensor(self.kernel_generator.kernel_volume, in_channels, out_channels))
        self._function = me.MinkowskiConvolutionFunction()

    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> torch.Tensor:
        return self._function.apply(input.feats, self.kernel, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, input.coords_key, out_coords_key, input.coords_man)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel(self) -> torch.nn.Parameter:
        return self._kernel


class _WrappedMinkowskiConvolutionTranspose(_MinkowskiOperationWrapper):
    def __init__(self, in_channels: int, out_channels: int, output_padding: _size_any_t, **kwargs) -> None:
        super(_WrappedMinkowskiConvolutionTranspose, self).__init__(transposed=True, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._output_padding = torch.as_tensor(output_padding, dtype=torch.int32)  #TODO Como lidar co output_padding durante a avaliação do módulo?
        self._kernel = torch.nn.Parameter(torch.FloatTensor(self.kernel_generator.kernel_volume, in_channels, out_channels))
        self._function = me.MinkowskiConvolutionTransposeFunction()

    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> torch.Tensor:
        return self._function.apply(input.feats, self.kernel, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def output_padding(self) -> torch.IntTensor:
        return self._output_padding

    @property
    def kernel(self) -> torch.nn.Parameter:
        return self._kernel


class ConvNd(ConformalModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 stride: _size_any_t,
                 padding: _size_any_t,
                 dilation: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(ConvNd, self).__init__(name)
        self._native = _WrappedMinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, dilation={*map(int, self.dilation),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return self.out_channels, tuple(map(int, (torch.as_tensor(in_volume) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))

    def _register_parent(self, parent, index: int) -> None:
        parent.register_parameter(f'{self.__class__.__name__}[{index}]' if self._name is None else self._name, self._native.kernel)
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
            name=name)


class ConvTransposeNd(ConformalModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 stride: _size_any_t,
                 padding: _size_any_t,
                 output_padding: _size_any_t,
                 dilation: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(ConvTransposeNd, self).__init__(name)
        self._native = _WrappedMinkowskiConvolutionTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation)

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, output_padding={*map(int, self.output_padding),}, dilation={*map(int, self.dilation),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return self.out_channels, tuple((torch.as_tensor(in_volume) - 1) * self.stride - 2 * self.padding + self.output_padding + self.dilation * (self.kernel_size - 1) + 1)

    def _register_parent(self, parent, index: int) -> None:
        parent.register_parameter(f'{self.__class__.__name__}[{index}]' if self._name is None else self._name, self._native.kernel)
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
    def output_padding(self) -> torch.IntTensor:
        return self._native.output_padding

    @property
    def dilation(self) -> torch.IntTensor:
        return self._native.dilation

    @property
    def kernel(self) -> torch.nn.Parameter:
        return self._native.kernel


class ConvTranspose1d(ConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_1_t,
                 stride: _int_or_size_1_t=1,
                 padding: _int_or_size_1_t=0,
                 output_padding: _int_or_size_1_t=0,
                 dilation: _int_or_size_1_t=1,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            output_padding=_single(output_padding),
            dilation=_single(dilation),
            name=name)


class ConvTranspose2d(ConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_2_t,
                 stride: _int_or_size_2_t=1,
                 padding: _int_or_size_2_t=0,
                 output_padding: _int_or_size_2_t=0,
                 dilation: _int_or_size_2_t=1,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            output_padding=_pair(output_padding),
            dilation=_pair(dilation),
            name=name)


class ConvTranspose3d(ConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _int_or_size_3_t,
                 stride: _int_or_size_3_t=1,
                 padding: _int_or_size_3_t=0,
                 output_padding: _int_or_size_3_t=0,
                 dilation: _int_or_size_3_t=1,
                 name: Optional[str]=None) -> None:
        super(ConvTranspose3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            output_padding=_triple(output_padding),
            dilation=_triple(dilation),
            name=name)
