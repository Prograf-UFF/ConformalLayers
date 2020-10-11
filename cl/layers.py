from .activation import BaseActivation, NoActivation, SRePro
from .decorator import sync, singleton
from .extension import IdentityMatrix, ZeroTensor
from .module import ConformalModule
from .utils import _size_any_t
from typing import Iterator, List, Tuple, Union
import MinkowskiEngine as me
import numpy, operator, threading, torch


_cached_signature_t = Tuple[Tuple[int, _size_any_t], Tuple[int, _size_any_t], torch.dtype]  # Format: ((in_channels, in_volume), (out_channels, out_volume), dtype)


@singleton
class _EyeFactory(object):
    _Lock = threading.Lock()

    def __init__(self):
        self._cache = dict()

    @sync(_Lock)
    def clear_cache(self):
        self._cache.clear()

    @sync(_Lock)
    def get(self, in_channels: int, in_volume: _size_any_t) -> me.SparseTensor:
        key = (in_channels, in_volume)
        eye = self._cache.get(key)
        if eye is None:
            in_entries = numpy.prod(in_volume)
            in_numel = in_channels * in_entries
            coords = torch.stack(torch.meshgrid(*map(lambda end: torch.arange(int(end), dtype=torch.int32), (in_channels, *in_volume))), dim=-1).view(-1, 1 + len(in_volume))
            coords[:, 0] = torch.arange(len(coords), dtype=torch.int32)
            feats = torch.zeros(in_numel, in_channels)
            for channel in range(in_channels):
                feats[channel*in_entries:(channel+1)*in_entries, channel] = 1
            eye = me.SparseTensor(coords=coords, feats=feats)
            self._cache[key] = eye
        return eye


EYE_FACTORY = _EyeFactory()


class ConformalLayers(torch.nn.Module):
    def __init__(self, *args: ConformalModule) -> None:
        super(ConformalLayers, self).__init__()
        # Keep conformal modules as is and track parameter's updates
        self._modules: List[ConformalModule] = [*args]
        for index, curr in enumerate(self._modules):
            curr._register_parent(self, index)
        # Break the sequence of operations into Conformal Layers
        start = 0
        self._sequentials: List[torch.nn.Sequential] = list()
        self._activations: List[BaseActivation] = list()
        for ind, curr in enumerate(self._modules):
            if isinstance(curr, BaseActivation):
                self._sequentials.append(torch.nn.Sequential(*map(lambda module: module._native, self._modules[start:ind])))
                self._activations.append(curr)
                start = ind + 1
        if start != len(self._modules):
            self._sequentials.append(torch.nn.Sequential(*map(lambda module: module._native, self._modules[start:])))
            self._activations.append(NoActivation())
        # Initialize cached data with null values
        self._valid_cache = False
        self._cached_signature: _cached_signature_t = None
        self._cached_matrix: Union[torch.Tensor, CustomTensor] = None
        self._cached_tensor: Union[torch.Tensor, CustomTensor] = None

    def __repr__(self) -> str:
        return f'ConformalLayers{*self._modules,}'

    def __getitem__(self, index: int) -> ConformalModule:
        return self._modules[index]

    def __iter__(self) -> Iterator[ConformalModule]:
        return iter(self._modules)
    
    def __len__(self) -> int:
        return len(self._modules)

    def _compute_sequential_matrix(self, in_channels: int, in_volume: _size_any_t, out_channels: int, out_volume: _size_any_t, sequential: torch.nn.Module) -> torch.Tensor:
        in_numel = in_channels * numpy.prod(in_volume)
        out_numel = out_channels * numpy.prod(out_volume)
        tensor = sequential(EYE_FACTORY.get(in_channels, in_volume))
        coords = tensor.coords.view(-1, 1 + len(out_volume), 1).expand(-1, -1, out_channels).permute(0, 2, 1)
        coords = torch.cat((coords, torch.empty((len(coords), out_channels, 1), dtype=torch.int32)), 2)
        for channel in range(out_channels):
            coords[:, channel, -1] = channel
        coords = coords.view(-1, len(out_volume) + 2)
        coords = numpy.unravel_index(numpy.ravel_multi_index(tuple(coords[:, dim] for dim in (0, -1, *range(1, coords.shape[1] - 1))), (in_numel, out_channels, *out_volume)), shape=(in_numel, out_numel))
        feats = tensor.feats.view(-1)
        matrix = torch.sparse_coo_tensor((coords[1], coords[0]), feats, size=(out_numel + 1, in_numel + 1), dtype=feats.dtype)
        matrix += torch.sparse_coo_tensor(((out_numel,), (in_numel,)), (1,), size=(out_numel + 1, in_numel + 1), dtype=feats.dtype)
        return matrix

    def _update_cache(self, in_channels: int, in_volume: _size_any_t, dtype: torch.dtype) -> _cached_signature_t:
        in_signature = (in_channels, in_volume)
        if not self._valid_cache or self._cached_signature[0] != in_signature:
            in_numel = in_channels * numpy.prod(in_volume)
            # Compute the tensor representation of operatons in each layer
            out_channels, out_volume = in_channels, in_volume
            if self.nlayers > 0:
                tensors = [None] * self.nlayers
                for layer, (sequential, activation) in enumerate(zip(self._sequentials, self._activations)):
                    # Compute the number of channels and volume of the resulting batch entries
                    for module in sequential:
                        out_channels, out_volume = module.output_size(out_channels, out_volume)
                    # Make tensor representations of the operations in the current layer
                    sequential_matrix = self._compute_sequential_matrix(in_channels, in_volume, out_channels, out_volume, sequential)
                    activation_matrix, activation_tensor = activation.to_tensor(sequential_matrix)
                    tensors[layer] = (sequential_matrix, activation_matrix, activation_tensor)
                    # Get ready for the next layer
                    in_channels, in_volume = out_channels, out_volume
                out_numel = out_channels * numpy.prod(out_volume)
                # Compute the backward cummulative product of sequential matrices and activation matrices
                backward_prod_matrices = [None] * self.nlayers
                backward_prod_matrix = IdentityMatrix(out_numel + 1, dtype=dtype)
                backward_prod_matrices[self.nlayers - 1] = backward_prod_matrix
                for layer, (sequential_matrix, activation_matrix, _) in zip(range(self.nlayers - 2, -1, -1), tensors[-1:-self.nlayers:-1]):
                    backward_prod_matrix = torch.matmul(backward_prod_matrix, torch.matmul(activation_matrix, sequential_matrix))
                    backward_prod_matrices[layer] = backward_prod_matrix
                # Compute the final matrix encoding the Conformal Layers
                sequential_matrix, activation_matrix, _ = tensors[0]
                cached_matrix = torch.matmul(backward_prod_matrix, torch.matmul(activation_matrix, sequential_matrix))
                cached_matrix = cached_matrix.coalesce() #TODO Precisa somar valores em coordenadas duplicadas?
                # Compute the final tensor encoding the Conformal Layers
                cached_tensor = torch.sparse_coo_tensor(size=(out_numel + 1, in_numel + 1, in_numel + 1), dtype=dtype)
                forward_prod_sequential_matrix = IdentityMatrix(in_numel + 1, dtype=dtype)
                for backward_prod_matrix, (sequential_matrix, _, activation_tensor) in zip(backward_prod_matrices, tensors):
                    forward_prod_sequential_matrix = torch.matmul(sequential_matrix, forward_prod_sequential_matrix)
                    torch.add(cached_tensor, torch.tensordot(backward_prod_matrix, torch.tensordot(torch.tensordot(forward_prod_sequential_matrix, activation_tensor, dims=((0,), (1,))), forward_prod_sequential_matrix, dims=1), dims=((1,), (1,))), cached_tensor)
                cached_tensor = cached_tensor.coalesce() #TODO Precisa somar valores em coordenadas duplicadas?
            else:
                cached_matrix = IdentityMatrix(in_numel + 1, dtype=dtype)
                cached_tensor = ZeroTensor((in_numel + 1, in_numel + 1, in_numel + 1), dtype=dtype)
            self._cached_matrix = cached_matrix
            self._cached_tensor = cached_tensor
            # Set cached data as valid
            self._valid_cache = True
            self._cached_signature = (in_signature, (out_channels, out_volume), dtype)

            # in_numel = in_channels * numpy.prod(in_volume)
            # # Compute the number of channels and volume of the resulting batch entries
            # out_channels, out_volume = in_channels, in_volume
            # for curr in self._modules:
            #     out_channels, out_volume = curr.output_size(out_channels, out_volume)
            # out_numel = out_channels * numpy.prod(out_volume)
            # # Create a sparse tensor that decomposes any input
            # eye = EYE_FACTORY.get(in_channels, in_volume)
            # # Apply the modules to the eye tensor and make the tensor representation of the complete operation
            # assert self.nlayers == 1 #TODO Implementar a operação completa
            # result = self._sequentials[-1](eye) #TODO Encapsular esse processo em uma função
            # res_coords = result.coords.view(-1, 1 + len(out_volume), 1).expand(-1, -1, out_channels).permute(0, 2, 1)
            # res_coords = torch.cat((res_coords, torch.empty((len(res_coords), out_channels, 1), dtype=torch.int32)), 2)
            # for channel in range(out_channels):
            #     res_coords[:, channel, -1] = channel
            # res_coords = res_coords.view(-1, len(out_volume) + 2)
            # res_feats = result.feats.view(-1)
            # matrix_coords = numpy.unravel_index(numpy.ravel_multi_index(tuple(res_coords[:, dim] for dim in (0, -1, *range(1, res_coords.shape[1] - 1))), (in_numel, out_channels, *out_volume)), shape=(in_numel, out_numel))
            # self._cached_matrix = torch.sparse_coo_tensor((matrix_coords[1], matrix_coords[0]), res_feats, size=(out_numel, in_numel), dtype=res_feats.dtype)
            # # Set cached data as valid
            # self._valid_cache = True
            # self._cached_signature = ((in_channels, in_volume), (out_channels, out_volume))
        return self._cached_signature

    def forward(self, input: torch.Tensor):
        batches, in_channels, *in_volume = input.shape
        # If necessary, update cached data
        _, (out_channels, out_volume), _ = self._update_cache(in_channels, tuple(in_volume), input.dtype)
        # Reshape the input as a matrix where each batch entry corresponds to a column 
        input_as_matrix = input.view(batches, -1).t()
        input_as_matrix = torch.cat((input_as_matrix, input_as_matrix.norm(dim=0, keepdim=True)), dim=0) #TODO É possível eviar a cópia de memória?
        # Apply the Conformal Layers
        temp = torch.tensordot(self._cached_tensor, input_as_matrix, dims=1)
        output_as_matrix = torch.matmul(torch.add(temp, self._cached_matrix, temp), input_as_matrix)
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
