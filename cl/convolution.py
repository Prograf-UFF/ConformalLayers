from .module import MinkowskiOperationWrapper, ConformalModule
from .utils import DenseTensor, SparseTensor, IntOrSize1, IntOrSize2, IntOrSize3, SizeAny, Pair, Single, Triple
from collections import OrderedDict
from typing import Optional, Tuple
import MinkowskiEngine as me
import math, torch


class WrappedMinkowskiConvolution(MinkowskiOperationWrapper):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(WrappedMinkowskiConvolution, self).__init__(transposed=False, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel = torch.nn.Parameter(torch.FloatTensor(self.kernel_generator.kernel_volume, in_channels, out_channels))
        self._function = me.MinkowskiConvolutionFunction()
        self.reset_parameters()
        
    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> DenseTensor:
        return self._function.apply(input.feats, self.kernel, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, input.coords_key, out_coords_key, input.coords_man)

    def output_size(self, in_channels: int, in_volume: SizeAny) -> Tuple[int, SizeAny]:
        return self.out_channels, tuple(map(int, (torch.as_tensor(in_volume, dtype=torch.int32, device='cpu') + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self._kernel, a=math.sqrt(5))

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel(self) -> torch.nn.Parameter:
        return self._kernel


class WrappedMinkowskiConvolutionTranspose(MinkowskiOperationWrapper):
    def __init__(self, in_channels: int, out_channels: int, output_padding: SizeAny, **kwargs) -> None:
        super(WrappedMinkowskiConvolutionTranspose, self).__init__(transposed=True, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._output_padding = torch.as_tensor(output_padding, dtype=torch.int32, device='cpu')
        self._kernel = torch.nn.Parameter(torch.FloatTensor(self.kernel_generator.kernel_volume, in_channels, out_channels))
        self._function = me.MinkowskiConvolutionTransposeFunction()
        self.reset_parameters()
        raise NotImplementedError() #TODO Como lidar com output_padding durante a avaliação do módulo?

    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> DenseTensor:
        return self._function.apply(input.feats, self.kernel, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)

    def output_size(self, in_channels: int, in_volume: SizeAny) -> Tuple[int, SizeAny]:
        return self.out_channels, tuple((torch.as_tensor(in_volume, dtype=torch.int32, device='cpu') - 1) * self.stride - 2 * self.padding + self.output_padding + self.dilation * (self.kernel_size - 1) + 1)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self._kernel, a=math.sqrt(5))

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
                 kernel_size: SizeAny,
                 stride: SizeAny,
                 padding: SizeAny,
                 dilation: SizeAny,
                 *, name: Optional[str]=None) -> None:
        super(ConvNd, self).__init__(
            WrappedMinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation),
            name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['in_channels'] = self.in_channels
        entries['out_channels'] = self.out_channels
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        entries['dilation'] = tuple(map(int, self.dilation))
        return entries

    def reset_parameters(self) -> None:
        self.native.reset_parameters()

    @property
    def in_channels(self) -> int:
        return self.native.in_channels

    @property
    def out_channels(self) -> int:
        return self.native.out_channels

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

    @property
    def kernel(self) -> torch.nn.Parameter:
        return self.native.kernel


class Conv1d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntOrSize1,
                 stride: IntOrSize1=1,
                 padding: IntOrSize1=0,
                 dilation: IntOrSize1=1,
                 *, name: Optional[str]=None) -> None:
        super(Conv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=Single(kernel_size),
            stride=Single(stride),
            padding=Single(padding),
            dilation=Single(dilation),
            name=name)


class Conv2d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntOrSize2,
                 stride: IntOrSize2=1,
                 padding: IntOrSize2=0,
                 dilation: IntOrSize2=1,
                 *, name: Optional[str]=None) -> None:
        super(Conv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=Pair(kernel_size),
            stride=Pair(stride),
            padding=Pair(padding),
            dilation=Pair(dilation),
            name=name)


class Conv3d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntOrSize3,
                 stride: IntOrSize3=1,
                 padding: IntOrSize3=0,
                 dilation: IntOrSize3=1,
                 *, name: Optional[str]=None) -> None:
        super(Conv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=Triple(kernel_size),
            stride=Triple(stride),
            padding=Triple(padding),
            dilation=Triple(dilation),
            name=name)


class ConvTransposeNd(ConformalModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: SizeAny,
                 stride: SizeAny,
                 padding: SizeAny,
                 output_padding: SizeAny,
                 dilation: SizeAny,
                 *, name: Optional[str]=None) -> None:
        super(ConvTransposeNd, self).__init__(
            WrappedMinkowskiConvolutionTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation),
            name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['in_channels'] = self.in_channels
        entries['out_channels'] = self.out_channels
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        entries['output_padding'] = tuple(map(int, self.output_padding))
        entries['dilation'] = tuple(map(int, self.dilation))
        return entries

    def reset_parameters(self) -> None:
        self.native.reset_parameters()

    @property
    def in_channels(self) -> int:
        return self.native.in_channels

    @property
    def out_channels(self) -> int:
        return self.native.out_channels

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
    def output_padding(self) -> torch.IntTensor:
        return self.native.output_padding

    @property
    def dilation(self) -> torch.IntTensor:
        return self.native.dilation

    @property
    def kernel(self) -> torch.nn.Parameter:
        return self.native.kernel


class ConvTranspose1d(ConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntOrSize1,
                 stride: IntOrSize1=1,
                 padding: IntOrSize1=0,
                 output_padding: IntOrSize1=0,
                 dilation: IntOrSize1=1,
                 *, name: Optional[str]=None) -> None:
        super(ConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=Single(kernel_size),
            stride=Single(stride),
            padding=Single(padding),
            output_padding=Single(output_padding),
            dilation=Single(dilation),
            name=name)


class ConvTranspose2d(ConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntOrSize2,
                 stride: IntOrSize2=1,
                 padding: IntOrSize2=0,
                 output_padding: IntOrSize2=0,
                 dilation: IntOrSize2=1,
                 *, name: Optional[str]=None) -> None:
        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=Pair(kernel_size),
            stride=Pair(stride),
            padding=Pair(padding),
            output_padding=Pair(output_padding),
            dilation=Pair(dilation),
            name=name)


class ConvTranspose3d(ConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: IntOrSize3,
                 stride: IntOrSize3=1,
                 padding: IntOrSize3=0,
                 output_padding: IntOrSize3=0,
                 dilation: IntOrSize3=1,
                 *, name: Optional[str]=None) -> None:
        super(ConvTranspose3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=Triple(kernel_size),
            stride=Triple(stride),
            padding=Triple(padding),
            output_padding=Triple(output_padding),
            dilation=Triple(dilation),
            name=name)
