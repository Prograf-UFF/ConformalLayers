from .activation import BaseActivation, NoActivation, SRePro
from .decorator import sync, singleton
from .extension import IdentityMatrix, SparseTensor, ZeroTensor
from .module import ConformalModule
from .utils import SizeAny, ravel_multi_index, unravel_index
from collections import OrderedDict
from typing import Iterator, List, Tuple, Union
import MinkowskiEngine as me
import numpy, operator, threading, torch


_cached_signature_t = Tuple[Tuple[int, SizeAny], Tuple[int, SizeAny], torch.dtype]  # Format: ((in_channels, in_volume), (out_channels, out_volume), dtype)


@singleton
class EyeFactory(object):
    _Lock = threading.Lock()

    @sync(_Lock)
    def __init__(self) -> None:
        self._cache = dict()

    @sync(_Lock)
    def clear_cache(self):
        self._cache.clear()

    @sync(_Lock)
    def get(self, in_channels: int, in_volume: SizeAny, device: torch.device) -> me.SparseTensor:
        key = (in_channels, in_volume, device)
        eye = self._cache.get(key)
        if eye is None:
            in_entries = numpy.prod(in_volume)
            in_numel = in_channels * in_entries
            coords = torch.stack(torch.meshgrid(*map(lambda end: torch.arange(int(end), dtype=torch.int32), (in_channels, *in_volume))), dim=-1).view(-1, 1 + len(in_volume))
            coords[:, 0] = torch.arange(len(coords), dtype=torch.int32)
            feats = torch.zeros(in_numel, in_channels, device=device)
            for channel in range(in_channels):
                feats[channel*in_entries:(channel+1)*in_entries, channel] = 1
            eye = me.SparseTensor(feats, coords)
            self._cache[key] = eye
        return eye


EYE_FACTORY = EyeFactory()


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
        self._valid_cache = False
        self._cached_signature: _cached_signature_t = None
        self._cached_matrix: Union[SparseTensor, IdentityMatrix] = None
        self._cached_flat_tensor: Union[SparseTensor, ZeroTensor] = None

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

    def _compute_torch_module_matrix(self, in_channels: int, in_volume: SizeAny, out_channels: int, out_volume: SizeAny, module: torch.nn.Module, device: torch.device) -> SparseTensor:
        in_numel = in_channels * numpy.prod(in_volume)
        out_numel = out_channels * numpy.prod(out_volume)
        tensor = module(EYE_FACTORY.get(in_channels, in_volume, device))
        coords = tensor.coords.view(-1, 1 + len(out_volume), 1).expand(-1, -1, out_channels).permute(0, 2, 1)
        coords = torch.cat((coords, torch.empty((len(coords), out_channels, 1), dtype=torch.int32, device=coords.device)), 2)
        for channel in range(out_channels):
            coords[:, channel, -1] = channel
        coords = coords.view(-1, len(out_volume) + 2)
        row, col = unravel_index(ravel_multi_index(tuple(coords[:, dim] for dim in (0, -1, *range(1, coords.shape[1] - 1))), (in_numel, out_channels, *out_volume)), (in_numel, out_numel))
        row = torch.cat((row, torch.as_tensor((in_numel,), dtype=row.dtype, device=row.device))) #TODO Cópia de memória na concatenação
        col = torch.cat((col, torch.as_tensor((out_numel,), dtype=col.dtype, device=col.device))) #TODO Cópia de memória na concatenação
        values = tensor.feats.view(-1)
        values = torch.cat((values, torch.as_tensor((1,), dtype=values.dtype, device=values.device))) #TODO Cópia de memória na concatenação
        return SparseTensor(torch.stack((col, row,)), values, (out_numel + 1, in_numel + 1), coalesced=False)

    def _poor_mans_tensordots(self, activation_tensor: Union[SparseTensor, ZeroTensor], backward_prod_matrix: Union[SparseTensor, IdentityMatrix], forward_prod_sequential_matrix: SparseTensor) -> Union[SparseTensor, ZeroTensor]:
        A = activation_tensor               # Rank-3 tensor of size (s2, s2, s2)
        B = backward_prod_matrix            # Rank-2 tensor of size (s1, s2)
        C = forward_prod_sequential_matrix  # Rank-2 tensor of size (s2, s3)
                                            # output <- TensorDot[B, Transpose[TensorDot[TensorDot[Transpose[C, 0 <-> 1], Transpose[A]], C], 0 <-> 1]]
                                            #         = TensorDot[Transpose[TensorDot[B, TensorDot[Transpose[A, 1 <-> 2], C]], 1 <-> 2], C] is a rank-3 tensor of size (s1, s3, s3)
                                            # flat_output <- MatMul[FlattenLeft[Transpose[ReshapeRight[MatMul[B, FlattenRight[ReshapeLeft[MatMul[FlattenLeft[Transpose[A, 1 <-> 2]], C]]]]], 1 <-> 2]], C] is a rank-2 tensor of size (s1 * s3, s3)
        (s1, s2), (_, s3) = B.shape, C.shape
        # Check for trivial solution...
        if isinstance(A, ZeroTensor):
            return ZeroTensor((s1 * s3, s3), A.dtype, A.device)
        # ... otherwise, do it the hard way
        # Step 1: temp1 = FlattenLeft[Transpose[A, 1 <-> 2]]
        indices1 = unravel_index(ravel_multi_index((A.indices[0], A.indices[2], A.indices[1]), (s2, s2, s2)), (s2 * s2, s2))
        temp1 = SparseTensor(torch.stack(indices1), A.values, (s2 * s2, s2), coalesced=False)
        # Step 2: temp2 = MatMul(temp1, C)
        temp2 = torch.mm(temp1, C)
        # Step 3: temp3 = FlattenRight[ReshapeLeft[temp2]]
        indices3 = unravel_index(ravel_multi_index((temp2.indices[0], temp2.indices[1]), (s2 * s2, s3)), (s2, s2 * s3))
        temp3 = SparseTensor(torch.stack(indices3), temp2.values, (s2, s2 * s3), coalesced=False)
        # Step 4: temp4 = MatMul(B, temp3)
        temp4 = torch.mm(B, temp3)
        # Step 5: temp5 = FlattenLeft[Transpose[ReshapeRight[temp4], 1 <-> 2]]
        indices5 = unravel_index(ravel_multi_index((temp4.indices[0], temp4.indices[1]), (s1, s2 * s3)), (s1, s2, s3))
        indices5 = unravel_index(ravel_multi_index((indices5[0], indices5[2], indices5[1]), (s1, s3, s2)), (s1 * s3, s2))
        temp5 = SparseTensor(torch.stack(indices5), temp4.values, (s1 * s3, s2), coalesced=False)
        # Step 6: flat_output = MatMul(temp5, C)
        return torch.mm(temp5, C)

    def _update_cache(self, in_channels: int, in_volume: SizeAny, dtype: torch.dtype, device: torch.device) -> _cached_signature_t:
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
                    sequential_matrix = self._compute_torch_module_matrix(in_channels, in_volume, out_channels, out_volume, sequential, device)
                    activation_matrix, activation_tensor = activation.to_tensor(sequential_matrix)
                    tensors[layer] = (sequential_matrix, activation_matrix, activation_tensor)
                    # Get ready for the next layer
                    in_channels, in_volume = out_channels, out_volume
                out_numel = out_channels * numpy.prod(out_volume)
                # Compute the backward cummulative product of sequential matrices and activation matrices
                backward_prod_matrices = [None] * self.nlayers
                backward_prod_matrix = IdentityMatrix(out_numel + 1, dtype=dtype, device=device)
                backward_prod_matrices[self.nlayers - 1] = backward_prod_matrix
                for layer, (sequential_matrix, activation_matrix, _) in zip(range(self.nlayers - 2, -1, -1), tensors[-1:-self.nlayers:-1]):
                    backward_prod_matrix = torch.mm(backward_prod_matrix, torch.mm(activation_matrix, sequential_matrix))
                    backward_prod_matrices[layer] = backward_prod_matrix
                # Compute the final matrix encoding the Conformal Layers
                sequential_matrix, activation_matrix, _ = tensors[0]
                cached_matrix = torch.mm(backward_prod_matrix, torch.mm(activation_matrix, sequential_matrix))
                # Compute the final tensor encoding the Conformal Layers
                cached_flat_tensor = ZeroTensor(((out_numel + 1) * (in_numel + 1), in_numel + 1), dtype=dtype, device=device)
                forward_prod_sequential_matrix = IdentityMatrix(in_numel + 1, dtype=dtype, device=device)
                for backward_prod_matrix, (sequential_matrix, _, activation_tensor) in zip(backward_prod_matrices, tensors):
                    forward_prod_sequential_matrix = torch.mm(sequential_matrix, forward_prod_sequential_matrix)
                    cached_flat_tensor = torch.add(cached_flat_tensor, self._poor_mans_tensordots(activation_tensor, backward_prod_matrix, forward_prod_sequential_matrix))
            else:
                cached_matrix = IdentityMatrix(in_numel + 1, dtype=dtype, device=device)
                cached_flat_tensor = ZeroTensor(((in_numel + 1) * (in_numel + 1), in_numel + 1), dtype=dtype, device=device)
            self._cached_matrix = cached_matrix
            self._cached_flat_tensor = cached_flat_tensor
            # Set cached data as valid
            self._valid_cache = True
            self._cached_signature = (in_signature, (out_channels, out_volume), dtype)
        return self._cached_signature

    def forward(self, input: torch.Tensor):
        batches, in_channels, *in_volume = input.shape
        # If necessary, update cached data
        _, (out_channels, out_volume), _ = self._update_cache(in_channels, tuple(in_volume), input.dtype, input.device)
        # Reshape the input as a matrix where each batch entry corresponds to a column 
        input_as_matrix = input.view(batches, -1).t()
        input_as_matrix = torch.cat((input_as_matrix, input_as_matrix.norm(dim=0, keepdim=True)), dim=0) #TODO É possível eviar a cópia de memória?
        # Apply the Conformal Layers
        output_as_matrix = torch.add(
            torch.matmul(self._cached_matrix, input_as_matrix),
            torch.matmul( #TODO Essas multiplicações são muito custosa. Deve ser possível economizar, pois só a última fatia do tensor de rank 3 não é igual a zero.
                torch.matmul(
                    self._cached_flat_tensor,
                    input_as_matrix
                ).view(*self._cached_matrix.shape, batches).permute(2, 0, 1),
                input_as_matrix.t().view(batches, -1, 1)
            ).view(batches, -1).t())
        output_as_matrix[:-1, :] /= output_as_matrix[-1, :]
        # Reshape the result to the expected format
        return output_as_matrix[:-1, :].t().view(batches, out_channels, *out_volume)

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
