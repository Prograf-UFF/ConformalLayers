from .activation import BaseActivation, NoActivation, SRePro
from .module import ConformalModule
from .utils import DenseTensor, ScalarTensor, SparseTensor, SizeAny, ravel_multi_index, unravel_index
from collections import OrderedDict
from typing import Iterator, List, Tuple, Union
import MinkowskiEngine as me
import numpy, operator, threading, torch, types


CachedSignature = Tuple[Tuple[int, SizeAny], Tuple[int, SizeAny], torch.dtype, str]  # Format: ((in_channels, in_volume), (out_channels, out_volume), dtype, device_str)


class ConformalLayers(torch.nn.Module):
    def __init__(self, *args: ConformalModule) -> None:
        super(ConformalLayers, self).__init__()
        # Keep conformal modules as is and track parameter's updates
        self._modulez = torch.nn.ModuleList(args)
        # Break the sequence of operations into Conformal Layers
        start = 0
        self._sequentials: List[torch.nn.Sequential] = list()
        self._activations: List[BaseActivation] = list()
        for ind, curr in enumerate(self._modulez):
            if isinstance(curr, BaseActivation):
                self._sequentials.append(torch.nn.Sequential(*map(lambda module: module.native, self._modulez[start:ind])))
                self._activations.append(curr)
                start = ind + 1
        if start != len(self._modulez):
            self._sequentials.append(torch.nn.Sequential(*map(lambda module: module.native, self._modulez[start:])))
            self._activations.append(NoActivation())
        # Initialize cached data with null values
        self._cached_signature: CachedSignature = None
        self._cached_matrix: SparseTensor = None  # Euclidean space matrix
        self._cached_matrix_extra: ScalarTensor = None  # Homogeneous coordinate scalar
        self._cached_tensor_extra: SparseTensor = None  # The Euclidean space matrix at the last slice of the rank-3 tensor

    def __getitem__(self, index: int) -> ConformalModule:
        return self._modulez[index]

    def __iter__(self) -> Iterator[ConformalModule]:
        return iter(self._modulez)
    
    def __len__(self) -> int:
        return len(self._modulez)

    def __repr__(self) -> str:
        if len(self._modulez) != 0:
            modules_str = ',\n    '.join(map(lambda pair: f'{pair[0]}: {pair[1]}', enumerate(self._modulez)))
            return f'{self.__class__.__name__}(\n    {modules_str})'
        return f'{self.__class__.__name__}( ¯\_(ツ)_/¯ )'

    def _from_minkowski_tensor_to_sparse_matrix(self, tensor: me.SparseTensor, in_numel: int, out_channels: int, out_volume: SizeAny) -> SparseTensor:
        out_numel = out_channels * numpy.prod(out_volume)
        coords = tensor.coords.view(-1, 1 + len(out_volume), 1).expand(-1, -1, out_channels).permute(0, 2, 1)
        coords = torch.cat((coords, torch.empty((len(coords), out_channels, 1), dtype=torch.int32, device=coords.device)), 2)
        for channel in range(out_channels):
            coords[:, channel, -1] = channel
        coords = coords.view(-1, len(out_volume) + 2)
        row, col = unravel_index(ravel_multi_index(tuple(coords[:, dim] for dim in (0, -1, *range(1, coords.shape[1] - 1))), (in_numel, out_channels, *out_volume)), (in_numel, out_numel))
        values = tensor.feats.view(-1)
        return torch.sparse_coo_tensor(torch.stack((col, row,)), values, (out_numel, in_numel), device=values.device)

    def _make_eye_input(self, in_channels: int, in_volume: SizeAny, device: torch.device) -> me.SparseTensor:
        in_entries = numpy.prod(in_volume)
        in_numel = in_channels * in_entries
        coords = torch.stack(torch.meshgrid(*map(lambda end: torch.arange(int(end), dtype=torch.int32, device='cpu'), (in_channels, *in_volume))), dim=-1).view(-1, 1 + len(in_volume))
        coords[:, 0] = torch.arange(len(coords), dtype=torch.int32, device='cpu')
        feats = torch.zeros(in_numel, in_channels, device=device)
        for channel in range(in_channels):
            feats[channel*in_entries:(channel+1)*in_entries, channel] = 1
        return me.SparseTensor(feats, coords)
    
    def _make_identity_matrix(self, size: int, dtype: torch.dtype, device: torch.device) -> SparseTensor:
        ind = torch.arange(size, dtype=torch.int64, device=device)
        return torch.sparse_coo_tensor(torch.stack((ind, ind,)), torch.ones((size,), dtype=dtype, device=device), (size, size), device=device)

    def _make_zero_matrix(self, size: int, dtype: torch.dtype, device: torch.device) -> SparseTensor:
        return torch.sparse_coo_tensor(torch.empty((2, 0,), dtype=torch.int64, device=device), torch.empty((0,), dtype=dtype, device=device), (size, size), device=device)

    def _update_cache(self, in_channels: int, in_volume: SizeAny, dtype: torch.dtype, device: torch.device) -> CachedSignature:
        in_signature = (in_channels, in_volume)
        if self._cached_signature is None or self._cached_signature[0] != in_signature or self._cached_signature[2] != dtype or self._cached_signature[3] != str(device):
            # Compute the tensor representation of operatons in each layer
            in_numel = in_channels * numpy.prod(in_volume)
            out_channels, out_volume = in_channels, in_volume
            # Initialize the resulting variables
            cached_matrix = self._make_identity_matrix(in_numel, dtype=dtype, device=device)
            cached_matrix_extra = torch.as_tensor(1, dtype=dtype, device=device)
            cached_tensor_extra = self._make_zero_matrix(in_numel, dtype=dtype, device=device)
            # Compute the Euclidean portion of the product U^{layer} . ... . U^{2} . U^{1} and the scalar values of the activation functions for all values of 'layers'
            stored_layer_values: List[Tuple[SparseTensor, ScalarTensor, ScalarTensor]] = [None] * self.nlayers
            for layer, (sequential, activation) in enumerate(zip(self._sequentials, self._activations)):
                # Compute the eye tensor for the current layer
                eye = self._make_eye_input(out_channels, out_volume, device)
                # Compute the number of channels and volume of the output of this layer
                for module in sequential:
                    out_channels, out_volume = module.output_size(out_channels, out_volume)
                # Compute the sparse Minkowski tensor and the custom sparse matrix (tensor) representation of the Euclidean portion o sequential matrix U^{layer}
                sequential_matrix_me: me.SparseTensor = sequential(eye)
                sequential_matrix = self._from_minkowski_tensor_to_sparse_matrix(sequential_matrix_me, len(eye.coords), out_channels, out_volume)
                # Compute the Euclidean portion of the matrix product U^{layer} . ... . U^{2} . U^{1}
                cached_matrix = torch.sparse.mm(sequential_matrix, cached_matrix)
                # Compute the scalar values used to define the activation matrix M^{layer} and the activation rank-3 tensor T^{layer}
                activation_matrix_scalar, activation_tensor_scalar = activation.to_tensor(sequential_matrix)
                # Store computed values
                stored_layer_values[layer] = cached_matrix, activation_matrix_scalar, activation_tensor_scalar
            # Use the stored layer values to compute the extra component of cached matrix and the extra slice of the cached tensor
            for sequential_matrix_prod, activation_matrix_scalar, activation_tensor_scalar in reversed(stored_layer_values):
                if activation_tensor_scalar is not None:
                    cached_tensor_extra = cached_tensor_extra + torch.sparse.mm(sequential_matrix_prod.t(), sequential_matrix_prod) * (cached_matrix_extra * activation_tensor_scalar)
                if activation_matrix_scalar is not None:
                    cached_matrix_extra = cached_matrix_extra * activation_matrix_scalar
            # Set the final matrix and the final tensor encoding the Conformal Layers
            self._cached_matrix = cached_matrix
            self._cached_matrix_extra = cached_matrix_extra
            self._cached_tensor_extra = cached_tensor_extra
            ##[ConformalLayers Promise] Ensure that the grad of non-leaf tensors will be retained
            ##for data in self.cached_data():
            ##    if data.requires_grad:
            ##        data.retain_grad()
            # Set cached data as valid
            self._cached_signature = (in_signature, (out_channels, out_volume), dtype, str(device))
        return self._cached_signature

    def cached_data(self) -> Iterator[DenseTensor]:
        if self._cached_matrix is not None:
            yield self._cached_matrix.values
        if self._cached_matrix_extra is not None:
            yield self._cached_matrix_extra
        if self._cached_tensor_extra is not None:
            yield self._cached_tensor_extra.values
    
    def forward(self, input: DenseTensor):
        batches, in_channels, *in_volume = input.shape
        # If necessary, update cached data
        _, (out_channels, out_volume), _, _ = self._update_cache(in_channels, tuple(in_volume), input.dtype, input.device)
        # Reshape the input as a matrix where each batch entry corresponds to a column 
        input_as_matrix = input.view(batches, -1).t()
        input_as_matrix_extra = input_as_matrix.norm(dim=0, keepdim=True)
        # Apply the Conformal Layers
        output_as_matrix = torch.sparse.mm(self._cached_matrix, input_as_matrix)
        output_as_matrix_extra = self._cached_matrix_extra * input_as_matrix_extra
        output_as_matrix_extra = output_as_matrix_extra + (torch.sparse.mm(self._cached_tensor_extra, input_as_matrix) * input_as_matrix).sum(dim=0, keepdim=True)
        output_as_matrix = output_as_matrix / output_as_matrix_extra
        return output_as_matrix.t().view(batches, out_channels, *out_volume)

    def invalidate_cache(self) -> None:
        self._cached_signature = None
        me.clear_global_coords_man()  # When done using MinkowskiEngine for forward and backward, we must cleanup the coordinates manager

    @property
    def cached_signature(self) -> CachedSignature:
        return self._cached_signature

    @property
    def nlayers(self) -> int:
        return len(self._sequentials)

    @property
    def valid_cache(self) -> bool:
        return self._cached_signature is not None
