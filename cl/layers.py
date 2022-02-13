from .activation import BaseActivation, NoActivation
from .module import ConformalModule
from .utils import DenseTensor, ScalarTensor, SparseTensor, SizeAny, ravel_multi_index, unravel_index
from collections import namedtuple
from typing import Iterator, List, Optional, Tuple
import MinkowskiEngine as me
import numpy
import torch


InSignature = namedtuple('InSignature', ['dims', 'dtype', 'device_str'])
OutSignature = namedtuple('OutSignature', ['dims'])
CachedSignature = namedtuple('CachedSignature', ['in_signature', 'out_signature'])


class ConformalLayers(torch.nn.Module):
    
    def __init__(self, *args: ConformalModule, pruning_threshold: Optional[float] = 1e-5, keep_as_sparse : bool = False) -> None:
        super(ConformalLayers, self).__init__()
        self.pruning_threshold = float(pruning_threshold) if pruning_threshold is not None else None
        self.keep_as_sparse = keep_as_sparse
        # Keep conformal modules as is and track parameter's updates
        self._modulez = torch.nn.Sequential(*args)
        # Break the sequence of operations into Conformal Layers
        start = 0
        self._sequentials: List[torch.nn.Sequential] = list()
        self._activations: List[BaseActivation] = list()
        for ind, curr in enumerate(self._modulez):
            if isinstance(curr, BaseActivation):
                self._sequentials.append(self._modulez[start:ind])
                self._activations.append(curr)
                start = ind + 1
        if start != len(self._modulez):
            self._sequentials.append(self._modulez[start:])
            self._activations.append(NoActivation())
        # Initialize cached data with null values
        self._cached_signature: CachedSignature = None
        # Euclidean space matrix
        self._cached_matrix: SparseTensor = None
        # Homogeneous coordinate scalar
        self._cached_matrix_extra: ScalarTensor = None
        # The Euclidean space matrix at the last slice of the rank-3 tensor
        self._cached_tensor_extra: SparseTensor = None

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
        return f'{self.__class__.__name__}( ¯\\_(ツ)_/¯ )'

    def _from_minkowski_to_sparse_coo(self, tensor: me.SparseTensor, in_numel: int, out_dims: SizeAny) -> SparseTensor:
        out_channels, *out_volume = out_dims if len(out_dims) > 1 else (1, *out_dims)
        out_numel = numpy.prod(out_dims)
        coords = tensor.coordinates.view(-1, 1 + len(out_volume), 1).expand(-1, -1, out_channels).permute(0, 2, 1)
        coords = torch.cat((coords, torch.empty((len(coords), out_channels, 1), dtype=torch.int32, device=coords.device)), 2)
        for channel in range(out_channels):
            coords[:, channel, -1] = channel
        coords = coords.view(-1, len(out_volume) + 2)
        row, col = unravel_index(ravel_multi_index(tuple(coords[:, dim] for dim in (0, -1, *range(1, coords.shape[1] - 1))), (in_numel, out_channels, *out_volume)), (in_numel, out_numel))
        values = tensor.features.view(-1)
        T = torch.sparse_coo_tensor(torch.stack((col, row,)), values, (out_numel, in_numel), device=values.device)
        if not self.keep_as_sparse:
            T = T.to_dense()
        return T

    def _make_eye_input(self, in_dims: SizeAny, dtype: torch.dtype, device: torch.device) -> me.SparseTensor:
        in_channels, *in_volume = in_dims if len(in_dims) > 1 else (1, *in_dims)
        in_entries = numpy.prod(in_volume)
        in_numel = in_channels * in_entries
        coords = torch.stack(torch.meshgrid(*map(lambda end: torch.arange(int(end), dtype=torch.int32, device=device), (in_channels, *in_volume))), dim=-1).view(-1, 1 + len(in_volume))
        coords[:, 0] = torch.arange(len(coords), dtype=torch.int32, device=device)
        feats = torch.zeros(in_numel, in_channels, dtype=dtype, device=device)
        for channel in range(in_channels):
            feats[channel*in_entries:(channel+1)*in_entries, channel] = 1
        return me.SparseTensor(feats, coords)
    
    def _make_identity_matrix(self, size: int, dtype: torch.dtype, device: torch.device) -> SparseTensor:
        ind = torch.arange(size, dtype=torch.int64, device=device)
        T = torch.sparse_coo_tensor(torch.stack((ind, ind,)), torch.ones((size,), dtype=dtype, device=device), (size, size), device=device)
        if not self.keep_as_sparse:
            T = T.to_dense()
        return T

    def _make_zero_matrix(self, size: int, dtype: torch.dtype, device: torch.device) -> SparseTensor:
        T = torch.sparse_coo_tensor(torch.empty((2, 0,), dtype=torch.int64, device=device), torch.empty((0,), dtype=dtype, device=device), (size, size), device=device)
        if not self.keep_as_sparse:
            T = T.to_dense()
        return T

    def _prune_negligible_coefficients(self, tensor: SparseTensor) -> SparseTensor:
        mask = tensor.values().abs() >= self.pruning_threshold
        indices = tensor.indices()[:, mask]
        values = tensor.values()[mask]
        return torch.sparse_coo_tensor(indices, values, tensor.shape, device=values.device)

    def _update_cache(self, in_dims: SizeAny, dtype: torch.dtype, device: torch.device) -> CachedSignature:
        in_signature = InSignature(in_dims, dtype, str(device))
        if self._cached_signature is None or self._cached_signature.in_signature != in_signature:
            in_numel = numpy.prod(in_dims)
            # Initialize the resulting variables
            cached_matrix = self._make_identity_matrix(in_numel, dtype=dtype, device=device)
            cached_matrix_extra = torch.as_tensor(1, dtype=dtype, device=device)
            cached_tensor_extra = self._make_zero_matrix(in_numel, dtype=dtype, device=device)
            # Compute the Euclidean portion of the product U^{layer} . ... . U^{2} . U^{1} and
            # the scalar values of the activation functions for all values of 'layers'
            stored_layer_values: List[Tuple[SparseTensor, ScalarTensor, ScalarTensor]] = [None] * self.nlayers
            for layer, (sequential, activation) in enumerate(zip(self._sequentials, self._activations)):
                # Compute the eye tensor for the current layer
                eye = self._make_eye_input(in_dims, dtype=cached_matrix.dtype, device=cached_matrix.device)
                alpha_upper = torch.as_tensor(1, dtype=dtype, device=device)
                # Compute the number of channels and volume of the output of this layer
                out_dims = in_dims
                for module in sequential:
                    out_dims = module.output_dims(*out_dims)
                # Compute the sparse Minkowski tensor and the custom sparse matrix (tensor) representation
                # of the Euclidean portion o sequential matrix U^{layer}
                sequential_matrix_me, alpha_upper = sequential((eye, alpha_upper))
                sequential_matrix = self._from_minkowski_to_sparse_coo(sequential_matrix_me, len(eye.coordinates), out_dims)
                # Compute the Euclidean portion of the matrix product U^{layer} . ... . U^{2} . U^{1}
                if self.keep_as_sparse:
                    cached_matrix = torch.sparse.mm(sequential_matrix, cached_matrix)
                else:
                    cached_matrix = torch.mm(sequential_matrix, cached_matrix)

                # Compute the scalar values used to define the activation matrix M^{layer}
                # and the activation rank-3 tensor T^{layer}
                activation_matrix_scalar, activation_tensor_scalar = activation.to_tensor(alpha_upper)
                # Store computed values
                stored_layer_values[layer] = cached_matrix, activation_matrix_scalar, activation_tensor_scalar
                in_dims = out_dims
            # Use the stored layer values to compute the extra component of cached matrix
            # and the extra slice of the cached tensor
            for sequential_matrix_prod, activation_matrix_scalar, activation_tensor_scalar in reversed(stored_layer_values):
                if activation_tensor_scalar is not None:
                    if self.keep_as_sparse:
                        cached_tensor_extra = cached_tensor_extra + torch.sparse.mm(sequential_matrix_prod.t(), sequential_matrix_prod) * (cached_matrix_extra * activation_tensor_scalar)
                    else:
                        cached_tensor_extra = cached_tensor_extra + torch.mm(sequential_matrix_prod.t(), sequential_matrix_prod) * (cached_matrix_extra * activation_tensor_scalar)
                if activation_matrix_scalar is not None:
                    cached_matrix_extra = cached_matrix_extra * activation_matrix_scalar
            if self.keep_as_sparse:
                cached_matrix = cached_matrix.coalesce()
                cached_tensor_extra = cached_tensor_extra.coalesce()
                # Remove negligible coefficients from sparse tensors
                if self.pruning_threshold is not None:
                    cached_matrix = self._prune_negligible_coefficients(cached_matrix)
                    cached_tensor_extra = self._prune_negligible_coefficients(cached_tensor_extra)
            # Set the final matrix and the final tensor encoding the Conformal Layers
            self._cached_matrix = cached_matrix
            self._cached_matrix_extra = cached_matrix_extra
            self._cached_tensor_extra = cached_tensor_extra
            self._cached_signature = CachedSignature(in_signature, OutSignature(out_dims,))
        return self._cached_signature

    def cached_data(self) -> Iterator[DenseTensor]:
        if self._cached_matrix is not None:
            yield self._cached_matrix.values
        if self._cached_matrix_extra is not None:
            yield self._cached_matrix_extra
        if self._cached_tensor_extra is not None:
            yield self._cached_tensor_extra.values
    
    def forward(self, input: DenseTensor) -> DenseTensor:
        batches, *in_dims = input.shape
        # Decide whether to use conventional processing or Conformal Layers-based processing
        if self.training:
            # If we are training, then the cache will not be valid for evaluation since the parameters will change
            self.invalidate_cache()
            # Apply the modules as is
            input_extra = torch.linalg.norm(input.view(batches, -1), ord=2, dim=1, keepdim=True)
            alpha_upper = torch.as_tensor(1, dtype=input.dtype, device=input.device)
            (output, output_extra), _ = self._modulez(((input, input_extra), alpha_upper))
            output = output / output_extra.view(batches, *map(lambda _: 1, range(1, output.dim())))
        else:
            with torch.no_grad():
                # Reshape the input as a matrix where each batch entry corresponds to a column 
                input_as_matrix = input.view(batches, -1).t()
                input_as_matrix_extra = torch.linalg.norm(input_as_matrix, ord=2, dim=0, keepdim=True)
                # If necessary, update cached data
                _, (out_dims,) = self._update_cache(tuple(in_dims), input.dtype, input.device)
                # Apply the modules using the compact representation of the Conformal Layers
                output_as_matrix = torch.mm(self._cached_matrix, input_as_matrix)
                output_as_matrix_extra = self._cached_matrix_extra * input_as_matrix_extra
                output_as_matrix_extra += (torch.mm(self._cached_tensor_extra, input_as_matrix) * input_as_matrix).sum(dim=0, keepdim=True)
                output_as_matrix /= output_as_matrix_extra
                # Reshape the output as expected
                output = output_as_matrix.t().view(batches, *out_dims)
        return output

    def invalidate_cache(self) -> None:
        self._cached_signature = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def cached_signature(self) -> Optional[CachedSignature]:
        return self._cached_signature

    @property
    def nlayers(self) -> int:
        return len(self._sequentials)

    @property
    def valid_cache(self) -> bool:
        return self._cached_signature is not None
