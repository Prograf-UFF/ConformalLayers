from .utils import _size_any_t
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import MinkowskiEngine as me
import torch


class _MinkowskiModuleWrapper(object):
    def __init__(self, module: me.MinkowskiModuleBase) -> None:
        self._module = module
        # Compute some constant values and keep them
        temp = self._module.kernel_size.sub(1).floor_divide(2).mul(self._module.kernel_size.remainder(2)).mul(self._module.dilation)
        padding = self._module.dilation * (self._module.kernel_size - 1) - self._module.padding if self._module.is_transpose else self._module.padding
        self._index_start_offset = temp.sub(padding)
        self._index_end_offset = temp.add(padding).sub(self._module.kernel_size.sub(1).mul(self._module.dilation)).add(1)

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        in_coords = input.coords
        min_in_coords = in_coords.min(0, keepdim=True)[0].view(-1)
        max_in_coords = in_coords.max(0, keepdim=True)[0].view(-1)
        # Compute the complete set of coordinates for evaluating the module
        indices = torch.stack(torch.meshgrid(*map(lambda start, end, step: torch.arange(int(start), int(end), int(step), dtype=torch.int32),
            min_in_coords[1:].add(self._index_start_offset),
            max_in_coords[1:].add(self._index_end_offset),
            self._module.stride
        )), dim=-1).view(-1, input.dimension)
        batches = torch.arange(min_in_coords[0], max_in_coords[0] + 1, dtype=torch.int32).view(-1, 1)
        # Evaluate the module considering only a subset of the complete set of coordinates per batch
        coords = torch.cat((
            batches.repeat_interleave(len(indices), dim=0),
            indices.repeat(max_in_coords[0] - min_in_coords[0] + 1, 1)  #TODO Nem todo indice precisa estar em todo batch
        ), dim=1)
        result = self._module._super_forward(input, coords)
        # Compress the resulting tensor coordinates
        if self._module.stride.ne(1).any():
            result = me.SparseTensor(
                coords=result.coords.sub(torch.cat((torch.zeros((1,), dtype=torch.int32), indices[0, :]))).floor_divide(torch.cat((torch.ones((1,), dtype=torch.int32), self._module.stride))),
                feats=result.feats
            )
        # Return the resulting tensor
        return result


class ConformalModule(ABC):
    def __init__(self, name: Optional[str]=None) -> None:
        super(ConformalModule, self).__init__()
        self._name = name

    def _extra_repr(self, comma: bool) -> str:
        return '' if self._name is None else f'{", " if comma else ""}name={self._name}'

    @abstractmethod
    def _output_size(self, in_channels: int, in_volume: _size_any_t) -> Tuple[int, _size_any_t]:
        pass
    
    def _register_parent(self, parent, index: int) -> None:
        pass

    @property
    def name(self) -> Optional[str]:
        return self._name
