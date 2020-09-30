from .activation import SRePro
from .module import ConformalModule
from .utils import _size_any_t
from typing import Iterator, Tuple
import MinkowskiEngine as me
import numpy, operator, torch


_cached_signature_t = Tuple[Tuple[int, _size_any_t], Tuple[int, _size_any_t]]  # Format: ((in_channels, in_volume), (out_channels, out_volume))


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
        # Initialize some useful attributes
        self._in_channels = None
        for curr in self._modules:
            if hasattr(curr, 'in_channels'):
                self._in_channels = curr.in_channels
                break
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
            # Create a sparse tensor that decomposes the input
            in_entries = numpy.prod(in_volume)
            def coords_generator():
                for batch, (_, *index) in enumerate(numpy.ndindex(in_channels, *in_volume)):
                    yield batch, *index
            eye = me.SparseTensor(
                coords=torch.from_numpy(numpy.fromiter(coords_generator(), dtype=[('', numpy.int32) for _ in range(1 + len(in_volume))]).view((numpy.int32, (1 + len(in_volume),)))),
                feats=torch.eye(in_channels).repeat_interleave(in_entries, dim=0)  #TODO É possível reaproveitar a memória criando view equivalente a repeat_interleave()?
            )
            # Apply the modules to the eye tensor and make the tensor representation of the complete operation
            #TODO Implementar a operação completa
            all_U = torch.nn.Sequential(*self._sequentials)
            result, _, _ = all_U(eye).dense() #TODO Precisa fazer a conversão para torch.Tensor?
            self._cached_left_tensor = result.view(in_channels * in_entries, -1).t() #TODO É possível reaproveitar a memória alocada anteriormente para o tensor?
            # Set cached data as valid
            self._valid_cache = True
            self._cached_signature = ((in_channels, in_volume), (out_channels, out_volume))
        return self._cached_signature

    def forward(self, input):
        batches, in_channels, *in_volume = input.shape
        in_volume = tuple(in_volume)
        if not self._in_channels is None and in_channels != self._in_channels:
            raise RuntimeError(f'Expected input to have {self._in_channels} channels, but got {in_channels} channels instead.')
        # If necessary, update cached data
        _, (out_channels, out_volume) = self._update_cache(in_channels, in_volume)
        # Reshape the input as a matrix where each batch entry corresponds to a column 
        x = input.view(batches, -1).t()
        # Apply the conformal layers
        y = self._cached_left_tensor.matmul(x) #TODO Implementar a operação completa
        # Reshape the result to the expected format
        return y.t().view(batches, out_channels, *out_volume)

    def invalidate_cache(self) -> None:
        self._valid_cache = False

    @property
    def cached_signature(self) -> _cached_signature_t:
        return self._cached_signature
        
    @property
    def valid_cache(self) -> bool:
        return self._valid_cache
