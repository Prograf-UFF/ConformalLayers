from .module import _MinkowskiModuleWrapper, ConformalModule
from .utils import _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from typing import Optional, Tuple
import MinkowskiEngine as me
import torch


class _WrappedMinkowskiConvolution(me.MinkowskiConvolution):
    def __init__(self, padding: _size_any_t, *args, **kwargs) -> None:
        super(_WrappedMinkowskiConvolution, self).__init__(*args, **kwargs)
        self._padding = torch.as_tensor(padding, dtype=torch.int32)
        # Compute some constant values and keep them
        kernel_origin = self.dilation * ((self.kernel_size - 1) // 2) * (self.kernel_size % 2)
        dilated_kernel_size = self.dilation * (self.kernel_size - 1) + 1
        self._kernel_start_offset = kernel_origin - dilated_kernel_size + 1
        self._kernel_end_offset = kernel_origin
        self._index_start_offset = kernel_origin - self.padding
        self._index_end_offset = kernel_origin + self.padding - dilated_kernel_size + 2

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        in_coords = input.coords
        batches = int(in_coords[:, 0].max()) + 1
        # Compute the complete set of coordinates for evaluating the module
        index_start = in_coords[:, 1:].min(0)[0] + self._index_start_offset
        index_end = in_coords[:, 1:].max(0)[0] + self._index_end_offset
        def cat_batch_and_indices(batch, indices):
            return torch.cat((torch.full((len(indices), 1), batch, dtype=torch.int32), indices), dim=1)
        def batch_coords(batch, indices):
            return cat_batch_and_indices(batch, torch.stack(torch.meshgrid(*map(lambda start, end, step: torch.arange(int(start), int(end), int(step), dtype=torch.int32),
                torch.max(index_start, ((indices.min(0)[0] + self._kernel_start_offset - index_start) // self.stride) * self.stride + index_start),
                torch.min(index_end, ((indices.max(0)[0] + self._kernel_end_offset - index_start) // self.stride + 1) * self.stride + index_start),
                self.stride)), dim=-1).view(-1, input.dimension))
        coords = torch.cat([batch_coords(batch, input.coordinates_at(batch)) for batch in range(batches)], dim=0)
        #TODO assert (torch.abs(result.feats) <= 1e-6).all(), 'Os limites do arange(...) precisam ser ajustados, pois coordenadas irrelevantes são geradas em casos a serem investigados
        # Evaluate the module
        result = super().forward(input, coords)
        # Map the first indices to zeros and compress the resulting coordinates
        if (index_start != 0).any():
            new_coords = result.coords
            new_coords[:, 1:] -= index_start
            if (self.stride != 1).any():
                new_coords[:, 1:] //= self.stride
                result = me.SparseTensor(coords=new_coords, feats=result.feats)
            else:
                result = me.SparseTensor(coords=new_coords, feats=result.feats)
        elif (self.stride != 1).any():
            new_coords = result.coords
            new_coords[:, 1:] //= self.stride
            result = me.SparseTensor(coords=new_coords, feats=result.feats)
        # Return the resulting tensor
        return result

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
                 name: Optional[str]=None) -> None:
        super(ConvNd, self).__init__(name)
        self._native = _WrappedMinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dimension=len(kernel_size),
            has_bias=False)

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, dilation={*map(int, self.dilation),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return self.out_channels, tuple(map(int, (torch.as_tensor(in_volume) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1))

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
                 dilation: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(ConvTransposeNd, self).__init__(name)
        self._native = _WrappedMinkowskiConvolutionTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dimension=len(kernel_size),
            has_bias=False)

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}, dilation={*map(int, self.dilation),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return self.out_channels, tuple((torch.as_tensor(in_volume) - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1)

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
    def kernel(self) -> torch.nn.Parameter:
        return self._native.kernel


class ConvTranspose1d(ConvTransposeNd):
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
            name=name)


class ConvTranspose2d(ConvTransposeNd):
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
            name=name)


class ConvTranspose3d(ConvTransposeNd):
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
            name=name)
