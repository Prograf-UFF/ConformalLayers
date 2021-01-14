from .module import ConformalModule, NativeModuleWrapper
from .utils import SizeAny, ravel_multi_index, unravel_index
from collections import OrderedDict
from typing import Optional
import MinkowskiEngine as me
import numpy, torch


class FlattenWrapper(NativeModuleWrapper):
    def __init__(self) -> None:
        super(FlattenWrapper, self).__init__()
        self._start_dim = 1 #TODO Implement start_dim != 1
        self._end_dim = -1 #TODO Implement end_dim != -1

    def forward(self, input: me.SparseTensor) -> me.SparseTensor:
        in_channels = input.feats.shape[1]

        in_volume = input.coords[:, 1:].max(0)[0] #TODO Esse max pode ser carregado módulo a módulo?
        in_volume += 1

        in_numel = in_channels * int(in_volume.prod())

        in_coords = input.coords.view(-1, 1 + len(in_volume), 1).expand(-1, -1, in_channels).permute(0, 2, 1)
        in_coords = torch.cat((in_coords, torch.empty((len(in_coords), in_channels, 1), dtype=torch.int32, device=in_coords.device)), 2)
        for channel in range(in_channels):
            in_coords[:, channel, -1] = channel
        in_coords = in_coords.view(-1, len(in_volume) + 2)
        
        out_coords = torch.stack(unravel_index(ravel_multi_index(tuple(in_coords[:, dim] for dim in (0, -1, *range(1, in_coords.shape[1] - 1))), (in_numel, in_channels, *in_volume)), (in_numel, in_numel))).t()
        out_feats = input.feats.view(-1, 1)
        
        return me.SparseTensor(out_feats, out_coords)

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (numpy.prod(in_dims),)

    @property
    def start_dim(self):
        return self._start_dim

    @property
    def end_dim(self):
        return self._end_dim


class Flatten(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(Flatten, self).__init__(FlattenWrapper(), name=name)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['start_dim'] = self.start_dim
        entries['end_dim'] = self.end_dim
        return entries

    @property
    def start_dim(self):
        return self.native.start_dim

    @property
    def end_dim(self):
        return self.native.end_dim
