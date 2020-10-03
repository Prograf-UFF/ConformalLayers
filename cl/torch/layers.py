from .activation import SRePro
from .module import ConformalModule
from .utils import _size_any_t
from typing import Iterator, Tuple
import MinkowskiEngine as me
import numpy, operator, torch


_cached_signature_t = Tuple[Tuple[int, _size_any_t], Tuple[int, _size_any_t]]  # Format: ((in_channels, in_volume), (out_channels, out_volume))


_union = me.MinkowskiUnion()


class ConformalLayers(torch.nn.Module):
    def __init__(self, *args: ConformalModule) -> None:
        super(ConformalLayers, self).__init__()
        # Keep conformal modules as is and track parameter's updates
        self._modules = [*args]
        for index, curr in enumerate(self._modules):
            curr._register_parent(self, index)
        # Break the sequence of operations into Conformal Layers
        start = 0
        self._sequentials = list()
        self._activations = list()
        for ind, curr in enumerate(self._modules):
            if isinstance(curr, SRePro):
                self._sequentials.append(torch.nn.Sequential(*map(lambda module: module._native, self._modules[start:ind])))
                self._activations.append(curr)
                start = ind + 1
        if start != len(self._modules):
            self._sequentials.append(torch.nn.Sequential(*map(lambda module: module._native, self._modules[start:])))
            self._activations.append(None) #TODO Lidar com esse caso
        # Initialize cached data with null values
        self._valid_cache = False
        self._cached_signature = None
        self._cached_left_tensor = None
        self._cached_right_tensor = None

    def __repr__(self) -> str:
        return f'ConformalLayers{*self._modules,}'

    def __getitem__(self, index: int) -> ConformalModule:
        return self._modules[index]

    def __iter__(self) -> Iterator[ConformalModule]:
        return iter(self._modules)
    
    def __len__(self) -> int:
        return len(self._modules)

    def _update_cache(self, in_channels: int, in_volume: _size_any_t) -> _cached_signature_t:
        if not self._valid_cache or self._cached_signature[0] != (in_channels, in_volume):
            # Compute the number of channels and volume of the resulting batch entries
            out_channels, out_volume = in_channels, in_volume
            for curr in self._modules:
                out_channels, out_volume = curr._output_size(out_channels, out_volume)
            # Create a sparse tensor that decomposes any input
            in_entries = numpy.prod(in_volume)
            in_numel = in_channels * in_entries
            coords = torch.stack(torch.meshgrid(*map(lambda end: torch.arange(int(end), dtype=torch.int32), (in_channels, *in_volume))), dim=-1).view(-1, 1 + len(in_volume))
            coords[:, 0] = torch.arange(len(coords))
            feats = torch.zeros(in_numel, in_channels)
            for channel in range(in_channels):
                feats[channel*in_entries:(channel+1)*in_entries, channel] = 1
            eye = me.SparseTensor(coords=coords, feats=feats)
            # Apply the modules to the eye tensor and make the tensor representation of the complete operation
            assert self.nlayers == 1 #TODO Implementar a operação completa
            min_coords = torch.zeros((1 + len(out_volume),), dtype=torch.int32)
            max_coords = torch.as_tensor((in_numel, *out_volume), dtype=torch.int32) - 1
            result = self._sequentials[-1](eye)
            bounds = me.SparseTensor(coords=torch.stack((min_coords, max_coords), dim=0), feats=torch.zeros((2, out_channels), dtype=result.dtype), coords_manager=result.coords_man, force_creation=True)
            result, _, _ = _union(result, bounds).dense() #TODO We use _union(...) as a workaround to avoid a bug in the dense(min_coords, max_coords) function.
            self._cached_left_tensor = result.view(in_numel, -1).t()
            # Set cached data as valid
            self._valid_cache = True
            self._cached_signature = ((in_channels, in_volume), (out_channels, out_volume))
        return self._cached_signature

    def forward(self, input):
        batches, in_channels, *in_volume = input.shape
        # If necessary, update cached data
        _, (out_channels, out_volume) = self._update_cache(in_channels, tuple(in_volume))
        # Reshape the input as a matrix where each batch entry corresponds to a column 
        input_as_matrix = input.view(batches, -1).t()
        # Apply the Conformal Layers
        output_as_matrix = self._cached_left_tensor.matmul(input_as_matrix) #TODO Implementar a operação completa
        # Reshape the result to the expected format
        return output_as_matrix.t().view(batches, out_channels, *out_volume)

    def invalidate_cache(self) -> None:
        self._valid_cache = False

    @property
    def cached_signature(self) -> _cached_signature_t:
        return self._cached_signature

    @property
    def nlayers(self) -> int:
        return len(self._sequentials)

    @property
    def valid_cache(self) -> bool:
        return self._valid_cache
