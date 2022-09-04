from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData, WrappedMinkowskiStridedOperation
from .utils import DenseTensor, ScalarTensor, IntOrSize1, IntOrSize2, IntOrSize3, Pair, Single, SizeAny, Triple
from collections import OrderedDict
from typing import Optional, Tuple, Union
import MinkowskiEngine as me
from MinkowskiEngineBackend._C import ConvolutionMode
import torch


class WrappedMinkowskiConvolution(WrappedMinkowskiStridedOperation):

    def __init__(self, owner: ConformalModule, kernel_generator: me.KernelGenerator) -> None:
        super(WrappedMinkowskiConvolution, self).__init__(owner)
        self._function = me.MinkowskiConvolutionFunction()
        self._kernel_generator = kernel_generator
        
    def _apply_function(self, input: me.SparseTensor, alpha_upper: ScalarTensor, out_coordinate_map_key: me.CoordinateMapKey) -> Tuple[DenseTensor, ScalarTensor]:
        # We don't know why Minkowski Engine convolution does not work with the view. It seems that the kernel must be contiguous.
        kernel = self.owner.weight.T.reshape(-1, self.owner.in_channels // self.owner.groups, self.owner.out_channels)
        # Handle channel grouping.
        if self.owner.groups == 1:
            # Compute the resulting feature.
            out_feats = self._function.apply(input.features, kernel.contiguous(), self._kernel_generator, ConvolutionMode.DEFAULT, input.coordinate_map_key, out_coordinate_map_key, input._manager)
        else:
            in_channels_step = self.owner.in_channels // self.owner.groups
            out_channels_step = self.owner.out_channels // self.owner.groups
            # Compute the resulting feature.
            out_feats = torch.cat([self._function.apply(input.features[:, in_channels_start:in_channels_start+in_channels_step], kernel[:, :, out_channels_start:out_channels_start+out_channels_step].contiguous(), self._kernel_generator, ConvolutionMode.DEFAULT, input.coordinate_map_key, out_coordinate_map_key, input._manager) for in_channels_start, out_channels_start in zip(range(0, self.owner.in_channels, in_channels_step), range(0, self.owner.out_channels, out_channels_step))], dim=1)
        # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
        alpha_upper = alpha_upper * torch.linalg.norm(self.owner.weight.view(self.owner.out_channels, self.owner.in_channels // self.owner.groups, -1), ord=1, dim=2).sum(dim=0).max()
        # Return results.
        return out_feats, alpha_upper


class ConvNd(ConformalModule):

    _TORCH_MODULE_CLASS = None

    def __init__(self, in_channels: int, out_channels: int, kernel_size: SizeAny, stride: SizeAny, padding: SizeAny, dilation: SizeAny, groups: int, *, name: Optional[str] = None) -> None:
        super(ConvNd, self).__init__(name=name)
        self._torch_module = self._TORCH_MODULE_CLASS(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode='zeros')
        self._minkowski_module = WrappedMinkowskiConvolution(self, me.KernelGenerator(kernel_size=kernel_size, stride=1, dilation=dilation, is_transpose=False, expand_coordinates=False, dimension=len(kernel_size)))

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['in_channels'] = self.in_channels
        entries['out_channels'] = self.out_channels
        entries['kernel_size'] = tuple(map(int, self.kernel_size))
        entries['stride'] = tuple(map(int, self.stride))
        entries['padding'] = tuple(map(int, self.padding))
        entries['dilation'] = tuple(map(int, self.dilation))
        entries['groups'] = self.groups
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            output = self._torch_module(input)
            # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
            alpha_upper = alpha_upper * torch.linalg.norm(self.weight.view(self.out_channels, self.in_channels, -1), ord=1, dim=2).sum(dim=0).max()
            return (output, input_extra), alpha_upper
        else:
            return self._minkowski_module(input)

    def output_dims(self, in_channels: int, *in_volume: int) -> SizeAny:
        return (self.out_channels, *map(int, torch.div(torch.as_tensor(in_volume, dtype=torch.int32, device=self.weight.device) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1, self.stride, rounding_mode='floor') + 1))

    @property
    def in_channels(self) -> int:
        return self._torch_module.in_channels

    @property
    def out_channels(self) -> int:
        return self._torch_module.out_channels

    @property
    def kernel_size(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.kernel_size, dtype=torch.int32, device=self.weight.device)

    @property
    def stride(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.stride, dtype=torch.int32, device=self.weight.device)

    @property
    def padding(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.padding, dtype=torch.int32, device=self.weight.device)

    @property
    def dilation(self) -> torch.IntTensor:
        return torch.as_tensor(self._torch_module.dilation, dtype=torch.int32, device=self.weight.device)

    @property
    def groups(self) -> int:
        return self._torch_module.groups

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._torch_module.weight


class Conv1d(ConvNd):

    _TORCH_MODULE_CLASS = torch.nn.Conv1d

    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrSize1, stride: IntOrSize1 = 1, padding: IntOrSize1 = 0, dilation: IntOrSize1 = 1, groups: int = 1, *, name: Optional[str] = None) -> None:
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=Single(kernel_size), stride=Single(stride), padding=Single(padding), dilation=Single(dilation), groups=groups, name=name)


class Conv2d(ConvNd):

    _TORCH_MODULE_CLASS = torch.nn.Conv2d

    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrSize2, stride: IntOrSize2 = 1, padding: IntOrSize2 = 0, dilation: IntOrSize2 = 1, groups: int = 1, *, name: Optional[str] = None) -> None:
        super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=Pair(kernel_size), stride=Pair(stride), padding=Pair(padding), dilation=Pair(dilation), groups=groups, name=name)


class Conv3d(ConvNd):
    
    _TORCH_MODULE_CLASS = torch.nn.Conv3d

    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrSize3, stride: IntOrSize3 = 1, padding: IntOrSize3 = 0, dilation: IntOrSize3 = 1, groups: int = 1, *, name: Optional[str] = None) -> None:
        super(Conv3d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=Triple(kernel_size), stride=Triple(stride), padding=Triple(padding), dilation=Triple(dilation), groups=groups, name=name)
