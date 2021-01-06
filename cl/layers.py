from .activation import BaseActivation, NoActivation, SRePro
from .extension import IdentityMatrix, SparseTensor, ZeroTensor
from .module import ConformalModule
from .utils import SizeAny, ravel_multi_index, unravel_index
from collections import OrderedDict
from typing import Iterator, List, Tuple, Union
import MinkowskiEngine as me
import numpy, operator, threading, torch, types


CachedSignature = Tuple[Tuple[int, SizeAny], Tuple[int, SizeAny], torch.dtype, str]  # Format: ((in_channels, in_volume), (out_channels, out_volume), dtype, device_str)


class ConformalLayers(torch.nn.Module):
    def __init__(self, *args: ConformalModule) -> None:
        super(ConformalLayers, self).__init__()
        # Keep conformal modules as is and track parameter's updates
        self._modulez: List[ConformalModule] = [*args]
        self._parameterz = torch.nn.ParameterList()
        for index, curr in enumerate(self._modulez):
            curr._register_parent(self, index)
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
        self._cached_matrix: Union[SparseTensor, IdentityMatrix] = None  # Euclidean space matrix
        self._cached_matrix_extra: torch.Tensor = None  # Homogeneous coordinate scalar
        self._cached_tensor_extra: Union[SparseTensor, ZeroTensor] = None  # The Euclidean space matrix at the last slice of the rank-3 tensor

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

    def _from_minkowski_tensor_to_custom_matrix(self, tensor: me.SparseTensor, in_numel: int, out_channels: int, out_volume: SizeAny) -> SparseTensor:
        out_numel = out_channels * numpy.prod(out_volume)
        coords = tensor.coords.view(-1, 1 + len(out_volume), 1).expand(-1, -1, out_channels).permute(0, 2, 1)
        coords = torch.cat((coords, torch.empty((len(coords), out_channels, 1), dtype=torch.int32, device=coords.device)), 2)
        for channel in range(out_channels):
            coords[:, channel, -1] = channel
        coords = coords.view(-1, len(out_volume) + 2)
        row, col = unravel_index(ravel_multi_index(tuple(coords[:, dim] for dim in (0, -1, *range(1, coords.shape[1] - 1))), (in_numel, out_channels, *out_volume)), (in_numel, out_numel))
        values = tensor.feats.view(-1)
        return SparseTensor(torch.stack((col, row,)), values, (out_numel, in_numel), coalesced=False)

    def _make_eye(self, in_channels: int, in_volume: SizeAny, device: torch.device) -> me.SparseTensor:
        in_entries = numpy.prod(in_volume)
        in_numel = in_channels * in_entries
        coords = torch.stack(torch.meshgrid(*map(lambda end: torch.arange(int(end), dtype=torch.int32, device='cpu'), (in_channels, *in_volume))), dim=-1).view(-1, 1 + len(in_volume))
        coords[:, 0] = torch.arange(len(coords), dtype=torch.int32, device='cpu')
        feats = torch.zeros(in_numel, in_channels, device=device)
        for channel in range(in_channels):
            feats[channel*in_entries:(channel+1)*in_entries, channel] = 1
        return me.SparseTensor(feats, coords)
    
    def _update_cache(self, in_channels: int, in_volume: SizeAny, dtype: torch.dtype, device: torch.device) -> CachedSignature:
        in_signature = (in_channels, in_volume)
        if self._cached_signature is None or self._cached_signature[0] != in_signature or self._cached_signature[2] != dtype or self._cached_signature[3] != str(device):
            # Compute the tensor representation of operatons in each layer
            in_numel = in_channels * numpy.prod(in_volume)
            out_channels, out_volume = in_channels, in_volume
            if self.nlayers > 0:
                # Compute the Euclidean portion of the product U^{layer} . ... . U^{2} . U^{1} and the scalar values of the activation functions for all values of 'layers'
                stored_layer_values: List[Tuple[SparseTensor, torch.Tensor, torch.Tensor]] = [None] * self.nlayers
                sequential_matrix_prod_me = self._make_eye(in_channels, in_volume, device)
                for layer, (sequential, activation) in enumerate(zip(self._sequentials, self._activations)):
                    # Compute the eye tensor for the current layer, except for the first one
                    eye = self._make_eye(out_channels, out_volume, device) if layer > 0 else sequential_matrix_prod_me  # Recall that at this point out_channels == in_channels and out_volume == in_volume for this layer
                    # Compute the number of channels and volume of the output of this layer
                    for module in sequential:
                        out_channels, out_volume = module.output_size(out_channels, out_volume)
                    # Compute the sparse Minkowski tensor and the custom sparse matrix (tensor) representation of the Euclidean portion o sequential matrix U^{layer}
                    sequential_matrix_me: me.SparseTensor = sequential(eye)
                    sequential_matrix = self._from_minkowski_tensor_to_custom_matrix(sequential_matrix_me, len(eye.coords), out_channels, out_volume)
                    # Compute the scalar values used to define the activation matrix M^{layer} and the activation rank-3 tensor T^{layer}
                    activation_matrix_scalar, activation_tensor_scalar = activation.to_tensor(sequential_matrix)  #TODO Existe uma multiplicação de matrizes esparsas aqui dentro
                    # Compute the sparse Minkowski tensor and the custom sparse matrix (tensor) representation of the Euclidean portion of the matrix product U^{layer} . ... . U^{2} . U^{1}
                    sequential_matrix_prod_me: me.SparseTensor = sequential(sequential_matrix_prod_me)
                    sequential_matrix_prod = self._from_minkowski_tensor_to_custom_matrix(sequential_matrix_prod_me, in_numel, out_channels, out_volume)
                    # Store computed values
                    stored_layer_values[layer] = sequential_matrix_prod, activation_matrix_scalar, activation_tensor_scalar
                # Use the stored layer values to compute the extra component of cached matrix and the extra slice of the cached tensor
                cached_matrix_extra = torch.as_tensor(1, dtype=dtype, device=device)
                cached_tensor_extra = ZeroTensor((in_numel, in_numel), dtype=dtype, device=device)
                for sequential_matrix_prod, activation_matrix_scalar, activation_tensor_scalar in reversed(stored_layer_values):
                    if activation_tensor_scalar is not None:
                        current_tensor_extra = torch.mm(sequential_matrix_prod.t(), sequential_matrix_prod) #TODO Multiplicação de matrizes esparsas
                        current_tensor_extra = torch.mul(current_tensor_extra, cached_matrix_extra * activation_tensor_scalar)
                        cached_tensor_extra = torch.add(cached_tensor_extra, current_tensor_extra)
                    cached_matrix_extra = cached_matrix_extra * activation_matrix_scalar
                # Set the final matrix and the final tensor encoding the Conformal Layers
                self._cached_matrix = stored_layer_values[-1][0]
                self._cached_matrix_extra = cached_matrix_extra
                self._cached_tensor_extra = cached_tensor_extra
                # Ensure that the grad of non-leaf tensors will be retained
                if not isinstance(self._cached_matrix, IdentityMatrix) and self._cached_matrix.values.requires_grad:
                    self._cached_matrix.values.retain_grad()
                if self._cached_matrix_extra.requires_grad:
                    self._cached_matrix_extra.retain_grad()
                if not isinstance(self._cached_tensor_extra, ZeroTensor) and self._cached_tensor_extra.values.requires_grad:
                    self._cached_tensor_extra.values.retain_grad()
            else:
                self._cached_matrix = IdentityMatrix(in_numel, dtype=dtype, device=device)
                self._cached_matrix_extra = torch.as_tensor(1, dtype=dtype, device=device)
                self._cached_tensor_extra = ZeroTensor((in_numel, in_numel), dtype=dtype, device=device)
            # Set cached data as valid
            self._cached_signature = (in_signature, (out_channels, out_volume), dtype, str(device))
        return self._cached_signature

    def cached_tensors(self) -> Iterator[torch.Tensor]:
        if self._cached_matrix is not None and not isinstance(self._cached_matrix, IdentityMatrix):
            yield self._cached_matrix.values
        if self._cached_matrix_extra is not None:
            yield self._cached_matrix_extra
        if self._cached_tensor_extra is not None and not isinstance(self._cached_tensor_extra, ZeroTensor):
            yield self._cached_tensor_extra.values
    
    def forward(self, input: torch.Tensor):
        batches, in_channels, *in_volume = input.shape
        # If necessary, update cached data
        _, (out_channels, out_volume), _, _ = self._update_cache(in_channels, tuple(in_volume), input.dtype, input.device)
        # Reshape the input as a matrix where each batch entry corresponds to a column 
        input_as_matrix = input.view(batches, -1).t()
        input_as_matrix_extra = input_as_matrix.norm(dim=0, keepdim=True)
        # Apply the Conformal Layers
        output_as_matrix = torch.mm(self._cached_matrix, input_as_matrix)
        output_as_matrix_extra = self._cached_matrix_extra * input_as_matrix_extra
        if not isinstance(self._cached_tensor_extra, ZeroTensor):
            output_as_matrix_extra = output_as_matrix_extra + (torch.mm(self._cached_tensor_extra, input_as_matrix) * input_as_matrix).sum(dim=0, keepdim=True)
        output_as_matrix = output_as_matrix / output_as_matrix_extra
        return output_as_matrix.t().view(batches, out_channels, *out_volume)

    def invalidate_cache(self) -> None:
        self._cached_signature = None

    @property
    def cached_signature(self) -> CachedSignature:
        return self._cached_signature

    @property
    def nlayers(self) -> int:
        return len(self._sequentials)

    @property
    def valid_cache(self) -> bool:
        return self._cached_signature is not None
