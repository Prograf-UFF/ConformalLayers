from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData, WrappedMinkowskiStridedOperation
from .utils import DenseTensor, ScalarTensor, IntOrSize1, IntOrSize2, IntOrSize3, Pair, Single, SizeAny, Triple
from collections import OrderedDict
from typing import Optional, Tuple, Union
import MinkowskiEngine as me
import torch


class WrappedMinkowskiConvolution(WrappedMinkowskiStridedOperation):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedMinkowskiConvolution, self).__init__(
            owner,
            transposed=False)
        self._function = me.MinkowskiConvolutionFunction()
        
    def _apply_function(self, input: me.SparseTensor, alpha_upper: ScalarTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> Tuple[DenseTensor, ScalarTensor]:
        kernel = torch.empty((self.kernel_generator.kernel_volume, self.owner.in_channels, self.owner.out_channels), dtype=self.owner.weight.dtype, device=self.owner.weight.device)
        kernel.copy_(self.owner.weight.T.reshape(*kernel.shape)) # We don't know why Minkowski Engine convolution does not work with the view
        out_feats = self._function.apply(input.feats, kernel, input.tensor_stride, 1, self.owner.kernel_size, self.owner.dilation, region_type, region_offset, input.coords_key, out_coords_key, input.coords_man)
        alpha_upper = alpha_upper * torch.linalg.norm(self.owner.weight.view(-1), ord=1) # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
        return out_feats, alpha_upper


class WrappedMinkowskiConvolutionTranspose(WrappedMinkowskiStridedOperation):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedMinkowskiConvolutionTranspose, self).__init__(
            owner,
            transposed=True)
        self._function = me.MinkowskiConvolutionTransposeFunction()
        raise NotImplementedError() #TODO Como lidar com output_padding durante a avaliação do módulo?

    def _apply_function(self, input: me.SparseTensor, alpha_upper: ScalarTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> Tuple[DenseTensor, ScalarTensor]:
        kernel = self.owner.weight.permute(1, 0, *range(2, self.owner.weight.dim())).T.view(self.kernel_generator.kernel_volume, self.owner.in_channels, self.owner.out_channels)
        out_feats = self._function.apply(input.feats, self.kernel, input.tensor_stride, 1, self.kernel_size, self.dilation, region_type, region_offset, False, input.coords_key, out_coords_key, input.coords_man)
        alpha_upper = alpha_upper * torch.linalg.norm(self.owner.weight.view(-1), ord=1) # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
        return out_feats, alpha_upper


class ConvNd(ConformalModule):
    _TORCH_MODULE_CLASS = None

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: SizeAny,
                 stride: SizeAny,
                 padding: SizeAny,
                 dilation: SizeAny,
                 *, name: Optional[str]=None) -> None:
        super(ConvNd, self).__init__(name=name)
        self._torch_module = self._TORCH_MODULE_CLASS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
            padding_mode='zeros')
        self._minkowski_module = WrappedMinkowskiConvolution(self)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['in_channels'] = self.in_channels
        entries['out_channels'] = self.out_channels
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        entries['dilation'] = tuple(map(int, self.dilation))
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            output = self._torch_module(input)
            alpha_upper = alpha_upper * torch.linalg.norm(self._torch_module.weight.view(-1), ord=1) # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
            return (output, input_extra), alpha_upper
        else:
            return self._minkowski_module(input)

    def output_dims(self, in_channels: int, *in_volume: int) -> SizeAny:
        return (self.out_channels, *map(int, (torch.as_tensor(in_volume, dtype=torch.int32, device='cpu') + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))

    @property
    def minkowski_module(self) -> torch.nn.Module:
        return self._minkowski_module

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._torch_module

    @property
    def in_channels(self) -> int:
        return self._torch_module.in_channels

    @property
    def out_channels(self) -> int:
        return self._torch_module.out_channels

    @property
    def kernel_size(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.kernel_size, dtype=torch.int32, device='cpu')

    @property
    def stride(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.stride, dtype=torch.int32, device='cpu')

    @property
    def padding(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.padding, dtype=torch.int32, device='cpu')

    @property
    def dilation(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.dilation, dtype=torch.int32, device='cpu')

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._torch_module.weight


class Conv1d(ConvNd):
    _TORCH_MODULE_CLASS = torch.nn.Conv1d

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
    _TORCH_MODULE_CLASS = torch.nn.Conv2d

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
    _TORCH_MODULE_CLASS = torch.nn.Conv3d

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
    _TORCH_MODULE_CLASS = None

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: SizeAny,
                 stride: SizeAny,
                 padding: SizeAny,
                 output_padding: SizeAny,
                 dilation: SizeAny,
                 *, name: Optional[str]=None) -> None:
        super(ConvTransposeNd, self).__init__(name=name)
        self._torch_module = self._TORCH_MODULE_CLASS(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=1,
            bias=False,
            padding_mode='zeros')
        self._minkowski_module = WrappedMinkowskiConvolutionTranspose(self)

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

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            output = self._torch_module(input)
            alpha_upper = alpha_upper * torch.linalg.norm(self._torch_module.weight.view(-1), ord=1) # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
            return (output, input_extra), alpha_upper
        else:
            return self._minkowski_module(input)

    def output_dims(self, in_channels: int, *in_volume: int) -> SizeAny:
        return (self.out_channels, *map(int, (torch.as_tensor(in_volume, dtype=torch.int32, device='cpu') - 1) * self.stride - 2 * self.padding + self.output_padding + self.dilation * (self.kernel_size - 1) + 1))

    @property
    def minkowski_module(self) -> torch.nn.Module:
        return self._minkowski_module

    @property
    def torch_module(self) -> torch.nn.Module:
        return self._torch_module

    @property
    def in_channels(self) -> int:
        return self._torch_module.in_channels

    @property
    def out_channels(self) -> int:
        return self._torch_module.out_channels

    @property
    def kernel_size(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.kernel_size, dtype=torch.int32, device='cpu')

    @property
    def stride(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.stride, dtype=torch.int32, device='cpu')

    @property
    def padding(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.padding, dtype=torch.int32, device='cpu')

    @property
    def output_padding(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.output_padding, dtype=torch.int32, device='cpu')

    @property
    def dilation(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.dilation, dtype=torch.int32, device='cpu')

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._torch_module.weight


class ConvTranspose1d(ConvTransposeNd):
    _TORCH_MODULE_CLASS = torch.nn.ConvTranspose1d

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
    _TORCH_MODULE_CLASS = torch.nn.ConvTranspose2d

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
    _TORCH_MODULE_CLASS = torch.nn.ConvTranspose3d

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
