from .module import _MinkowskiModuleWrapper, ConformalModule
from .utils import _int_or_size_1_t, _int_or_size_2_t, _int_or_size_3_t, _size_any_t, _pair, _single, _triple
from typing import Optional, Tuple
import MinkowskiEngine as me
import torch


class _WrappedMinkowskiAvgPooling(me.MinkowskiSumPooling):
    def __init__(self, stride: _size_any_t, padding: _size_any_t, *args, **kwargs) -> None:
        super(_WrappedMinkowskiAvgPooling, self).__init__(*args, **kwargs)
        self._wrapped_stride = torch.as_tensor(stride, dtype=torch.int32)
        self._padding = torch.as_tensor(padding, dtype=torch.int32)
        # Compute some constant values and keep them
        kernel_origin = self.dilation * ((self.kernel_size - 1) // 2) * (self.kernel_size % 2)
        dilated_kernel_size = self.dilation * (self.kernel_size - 1) + 1
        self._kernel_start_offset = kernel_origin - dilated_kernel_size + 1
        self._kernel_end_offset = kernel_origin
        self._index_start_offset = kernel_origin - self.padding
        self._index_end_offset = kernel_origin + self.padding - dilated_kernel_size + 2
        self._inv_cardinality = 1 / int(torch.prod(self.kernel_size))

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        in_coords = input.coords
        indices_per_batch = input.decomposed_coordinates
        # Compute the complete set of coordinates for evaluating the module
        index_start = in_coords[:, 1:].min(0)[0] + self._index_start_offset
        index_end = in_coords[:, 1:].max(0)[0] + self._index_end_offset
        out_coords = torch.cat(tuple(torch.stack(torch.meshgrid(torch.as_tensor((batch,), dtype=torch.int32), *map(lambda start, end, step: torch.arange(int(start), int(end), int(step), dtype=torch.int32),
            torch.max(index_start, ((indices.min(0)[0] + self._kernel_start_offset - index_start) // self.wrapped_stride) * self.wrapped_stride + index_start),
            torch.min(index_end, ((indices.max(0)[0] + self._kernel_end_offset - index_start) // self.wrapped_stride + 1) * self.wrapped_stride + index_start),
            self.wrapped_stride)), dim=-1).view(-1, 1 + input.dimension) for batch, indices in enumerate(indices_per_batch)), dim=0)
        #TODO assert (torch.abs(output.feats) <= 1e-6).all(), 'Os limites do arange(...) precisam ser ajustados, pois coordenadas irrelevantes são geradas em casos a serem investigados
        # Evaluate the module
        output = super().forward(input, out_coords)
        # Map the first indices to zeros, compress the resulting coordinates and divide the features by the kernel's cardinality
        new_coords = output.coords
        if (index_start != 0).any():
            new_coords[:, 1:] -= index_start
        if (self.wrapped_stride != 1).any():
            new_coords[:, 1:] //= self.wrapped_stride
        return me.SparseTensor(coords=new_coords, feats=output.feats * self._inv_cardinality)

    @property
    def wrapped_stride(self) -> torch.IntTensor:
        return self._wrapped_stride

    @property
    def padding(self) -> torch.IntTensor:
        return self._padding


class _WrappedMinkowskiSumPooling(me.MinkowskiSumPooling):
    def __init__(self, stride: _size_any_t, padding: _size_any_t, *args, **kwargs) -> None:
        super(_WrappedMinkowskiSumPooling, self).__init__(*args, **kwargs)
        self._wrapped_stride = torch.as_tensor(stride, dtype=torch.int32)
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
        indices_per_batch = input.decomposed_coordinates
        # Compute the complete set of coordinates for evaluating the module
        index_start = in_coords[:, 1:].min(0)[0] + self._index_start_offset
        index_end = in_coords[:, 1:].max(0)[0] + self._index_end_offset
        out_coords = torch.cat(tuple(torch.stack(torch.meshgrid(torch.as_tensor((batch,), dtype=torch.int32), *map(lambda start, end, step: torch.arange(int(start), int(end), int(step), dtype=torch.int32),
            torch.max(index_start, ((indices.min(0)[0] + self._kernel_start_offset - index_start) // self.wrapped_stride) * self.wrapped_stride + index_start),
            torch.min(index_end, ((indices.max(0)[0] + self._kernel_end_offset - index_start) // self.wrapped_stride + 1) * self.wrapped_stride + index_start),
            self.wrapped_stride)), dim=-1).view(-1, 1 + input.dimension) for batch, indices in enumerate(indices_per_batch)), dim=0)
        #TODO assert (torch.abs(output.feats) <= 1e-6).all(), 'Os limites do arange(...) precisam ser ajustados, pois coordenadas irrelevantes são geradas em casos a serem investigados
        # Evaluate the module
        output = super().forward(input, out_coords)
        # Map the first indices to zeros and compress the resulting coordinates
        if (index_start != 0).any():
            new_coords = output.coords
            new_coords[:, 1:] -= index_start
            if (self.wrapped_stride != 1).any():
                new_coords[:, 1:] //= self.wrapped_stride
            output = me.SparseTensor(coords=new_coords, feats=output.feats)
        elif (self.wrapped_stride != 1).any():
            new_coords = output.coords
            new_coords[:, 1:] //= self.wrapped_stride
            output = me.SparseTensor(coords=new_coords, feats=output.feats)
        # Return the resulting tensor
        return output

    @property
    def wrapped_stride(self) -> torch.IntTensor:
        return self._wrapped_stride

    @property
    def padding(self) -> torch.IntTensor:
        return self._padding


class AvgPoolNd(ConformalModule):
    def __init__(self,
                 kernel_size: _size_any_t,
                 stride: Optional[_size_any_t],
                 padding: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(AvgPoolNd, self).__init__(name)
        self._native = _WrappedMinkowskiAvgPooling(
            kernel_size=kernel_size,
            stride=kernel_size if stride is None else stride,
            padding=padding,
            dimension=len(kernel_size))

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, tuple(map(int, (torch.as_tensor(in_volume) + 2 * self.padding - self.kernel_size) // self.stride + 1))

    @property
    def kernel_size(self) -> torch.IntTensor:
        return self._native.kernel_size

    @property
    def stride(self) -> torch.IntTensor:
        return self._native.wrapped_stride

    @property
    def padding(self) -> torch.IntTensor:
        return self._native.padding

    @property
    def dimension(self) -> int:
        return self._native.dimension


class AvgPool1d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_1_t,
                 stride: Optional[_int_or_size_1_t]=None,
                 padding: _int_or_size_1_t=0,
                 name: Optional[str]=None) -> None:
        super(AvgPool1d, self).__init__(
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            name=name)


class AvgPool2d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_2_t,
                 stride: Optional[_int_or_size_2_t]=None,
                 padding: _int_or_size_2_t=0,
                 name: Optional[str]=None) -> None:
        super(AvgPool2d, self).__init__(
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            name=name)


class AvgPool3d(AvgPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_3_t,
                 stride: Optional[_int_or_size_3_t]=None,
                 padding: _int_or_size_3_t=0,
                 name: Optional[str]=None) -> None:
        super(AvgPool3d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            name=name)


class SumPoolNd(ConformalModule):
    def __init__(self,
                 kernel_size: _size_any_t,
                 stride: Optional[_size_any_t],
                 padding: _size_any_t,
                 name: Optional[str]=None) -> None:
        super(SumPoolNd, self).__init__(name)
        self._native = _WrappedMinkowskiSumPooling(
            kernel_size=kernel_size,
            stride=kernel_size if stride is None else stride,
            padding=padding,
            dimension=len(kernel_size))

    def __repr__(self) -> str:
       return f'{self.__class__.__name__}(kernel_size={*map(int, self.kernel_size),}, stride={*map(int, self.stride),}, padding={*map(int, self.padding),}{self._extra_repr(True)})'

    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        return in_channels, tuple(map(int, (torch.as_tensor(in_volume) + 2 * self.padding - self.kernel_size) // self.stride + 1))

    @property
    def kernel_size(self) -> torch.IntTensor:
        return self._native.kernel_size

    @property
    def stride(self) -> torch.IntTensor:
        return self._native.wrapped_stride

    @property
    def padding(self) -> torch.IntTensor:
        return self._native.padding

    @property
    def dimension(self) -> int:
        return self._native.dimension


class SumPool1d(SumPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_1_t,
                 stride: Optional[_int_or_size_1_t]=None,
                 padding: _int_or_size_1_t=0,
                 name: Optional[str]=None) -> None:
        super(SumPool1d, self).__init__(
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single(padding),
            name=name)


class SumPool2d(SumPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_2_t,
                 stride: Optional[_int_or_size_2_t]=None,
                 padding: _int_or_size_2_t=0,
                 name: Optional[str]=None) -> None:
        super(SumPool2d, self).__init__(
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair(padding),
            name=name)


class SumPool3d(SumPoolNd):
    def __init__(self,
                 kernel_size: _int_or_size_3_t,
                 stride: Optional[_int_or_size_3_t]=None,
                 padding: _int_or_size_3_t=0,
                 name: Optional[str]=None) -> None:
        super(SumPool3d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple(padding),
            name=name)
