from .utils import SizeAny
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple
import MinkowskiEngine as me
import torch

class NativeModuleWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super(NativeModuleWrapper, self).__init__()

    @abstractmethod
    def output_size(self, in_size: SizeAny) -> SizeAny:
        pass


class MinkowskiOperationWrapper(NativeModuleWrapper):
    def __init__(self,
                 kernel_size: SizeAny,
                 stride: SizeAny,
                 padding: SizeAny,
                 dilation: SizeAny,
                 transposed: bool) -> None:
        super(MinkowskiOperationWrapper, self).__init__()
        # Declare basic properties
        self._kernel_size = torch.as_tensor(kernel_size, dtype=torch.int32, device='cpu')
        self._stride = torch.as_tensor(stride, dtype=torch.int32, device='cpu')
        self._padding = torch.as_tensor(padding, dtype=torch.int32, device='cpu')
        self._dilation = torch.as_tensor(dilation, dtype=torch.int32, device='cpu')
        self._transposed = transposed
        self._kernel_generator = me.KernelGenerator(kernel_size=kernel_size, stride=1, dilation=dilation, dimension=len(kernel_size))
        # Compute some constant values and keep them
        kernel_origin = self.dilation * ((self.kernel_size - 1) // 2) * (self.kernel_size % 2)
        dilated_kernel_size = self.dilation * (self.kernel_size - 1) + 1
        self._kernel_start_offset = kernel_origin - dilated_kernel_size + 1
        self._kernel_end_offset = kernel_origin
        self._index_start_offset = kernel_origin - self.padding
        self._index_end_offset = kernel_origin + self.padding - dilated_kernel_size + 2

    @abstractmethod
    def _apply_function(self, input: me.SparseTensor, region_type: me.RegionType, region_offset: torch.IntTensor, out_coords_key: me.CoordsKey) -> torch.Tensor:
        pass

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        in_coords = input.coords
        indices_per_batch = input.decomposed_coordinates
        # Compute the complete set of coordinates for evaluating the module
        index_start = self._index_start_offset
        index_end = in_coords[:, 1:].max(0)[0] + self._index_end_offset #TODO Esse max pode ser reaproveitado dos max por batch?
        out_coords = torch.cat(tuple(torch.stack(torch.meshgrid(torch.as_tensor((batch,), dtype=torch.int32, device='cpu'), *map(lambda start, end, step: torch.arange(int(start), int(end), int(step), dtype=torch.int32, device='cpu'),
            torch.max(index_start, ((indices.min(0)[0] + self._kernel_start_offset - index_start) // self.stride) * self.stride + index_start),
            torch.min(index_end, ((indices.max(0)[0] + self._kernel_end_offset - index_start) // self.stride + 1) * self.stride + index_start),
            self.stride)), dim=-1).view(-1, 1 + input.dimension) for batch, indices in enumerate(indices_per_batch)), dim=0)
        #TODO assert (torch.abs(output.feats) <= 1e-6).all(), 'Os limites do arange(...) precisam ser ajustados, pois coordenadas irrelevantes são geradas em casos a serem investigados
        # Create a region_type, region_offset, and coords_key
        region_type, region_offset, _ = self._kernel_generator.get_kernel(input.tensor_stride, self.transposed)
        out_coords_key = input.coords_man.create_coords_key(out_coords, tensor_stride=1, force_creation=True, force_remap=True, allow_duplicate_coords=True)
        # Evaluate the module
        out_feats = self._apply_function(input, region_type, region_offset, out_coords_key)
        # Map the first indices to zeros and compress the resulting coordinates when needed
        if (index_start != 0).any():
            out_coords[:, 1:] -= index_start
            if (self.stride != 1).any():
                out_coords[:, 1:] //= self.stride
            return me.SparseTensor(out_feats, out_coords, coords_manager=input.coords_man, force_creation=True)
        elif (self.stride != 1).any():
            out_coords[:, 1:] //= self.stride
            return me.SparseTensor(out_feats, out_coords, coords_manager=input.coords_man, force_creation=True)
        else:
            return me.SparseTensor(out_feats, coords_key=out_coords_key, coords_manager=input.coords_man)

    @property
    def kernel_size(self) -> torch.IntTensor:
        return self._kernel_size

    @property
    def stride(self) -> torch.IntTensor:
        return self._stride

    @property
    def padding(self) -> torch.IntTensor:
        return self._padding

    @property
    def dilation(self) -> torch.IntTensor:
        return self._dilation

    @property
    def transposed(self) -> bool:
        return self._transposed

    @property
    def kernel_generator(self) -> me.KernelGenerator:
        return self._kernel_generator


class ConformalModule(ABC):
    def __init__(self,
                 native: NativeModuleWrapper,
                 *, name: Optional[str]=None) -> None:
        super(ConformalModule, self).__init__()
        self._native = native
        self._name = name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({", ".join(map(lambda pair: "{}={}".format(*pair), self._repr_dict().items()))})'

    def _repr_dict(self) -> OrderedDict:
        entries = OrderedDict()
        if self._name is not None:
            entries['name'] = self.name
        return entries

    def _register_parent(self, parent, index: int) -> None:
        pass

    def output_size(self, in_channels: int, in_volume: SizeAny) -> Tuple[int, SizeAny]:
        return self.native.output_size(in_channels, in_volume)
    
    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def native(self) -> NativeModuleWrapper:
        return self._native