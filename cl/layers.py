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


# @singleton
class EyeFactory(object):
    # _Lock = threading.Lock()

    # @sync(_Lock)
    def __init__(self) -> None:
        self._cache = dict()

    # @sync(_Lock)
    def clear_cache(self):
        self._cache.clear()

    # @sync(_Lock)
    def get(self, in_channels: int, in_volume: SizeAny, device: torch.device) -> me.SparseTensor:
        
        key = (in_channels, in_volume, device)
        eye = self._cache.get(key)
        if eye is None:
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            in_entries = numpy.prod(in_volume)
            in_numel = in_channels * in_entries
            coords = torch.stack(torch.meshgrid(*map(lambda end: torch.arange(int(end), dtype=torch.int32, device='cpu'), (in_channels, *in_volume))), dim=-1).view(-1, 1 + len(in_volume))
            coords[:, 0] = torch.arange(len(coords), dtype=torch.int32, device='cpu')
            feats = torch.zeros(in_numel, in_channels, device=device)
            for channel in range(in_channels):
                feats[channel*in_entries:(channel+1)*in_entries, channel] = 1
            eye = me.SparseTensor(feats, coords) 
            # end_event.record()
            self._cache[key] = eye
        return eye


EYE_FACTORY = EyeFactory()


class ConformalLayers(torch.nn.Module):
    def __init__(self, *args: ConformalModule) -> None:
        super(ConformalLayers, self).__init__()
        # Keep conformal modules as is and track parameter's updates
        self._modulez: List[ConformalModule] = [*args]
        self._parameterz = torch.nn.ParameterList()
        self._Lock = threading.Lock()
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
        self._cached_matrix: Union[SparseTensor, IdentityMatrix] = None  # Euclidean space matrix
        self._cached_matrix_extra: torch.Tensor = None  # Homogeneous coordinates scalar
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

    def _compute_torch_module_matrix(self, in_channels: int, in_volume: SizeAny, out_channels: int, out_volume: SizeAny, module: torch.nn.Module, device: torch.device) -> SparseTensor:
        in_numel = in_channels * numpy.prod(in_volume)
        out_numel = out_channels * numpy.prod(out_volume)
        eye = EyeFactory().get(in_channels, in_volume, device)
        tensor = module(eye)
        coords = tensor.coords.view(-1, 1 + len(out_volume), 1).expand(-1, -1, out_channels).permute(0, 2, 1)
        coords = torch.cat((coords, torch.empty((len(coords), out_channels, 1), dtype=torch.int32, device=coords.device)), 2)
        for channel in range(out_channels):
            coords[:, channel, -1] = channel
        coords = coords.view(-1, len(out_volume) + 2)
        row, col = unravel_index(ravel_multi_index(tuple(coords[:, dim] for dim in (0, -1, *range(1, coords.shape[1] - 1))), (in_numel, out_channels, *out_volume)), (in_numel, out_numel))
        values = tensor.feats.view(-1)
        return SparseTensor(torch.stack((col, row,)), values, (out_numel, in_numel), coalesced=False)

    @torch.no_grad()
    def _update_cache(self, in_channels: int, in_volume: SizeAny, dtype: torch.dtype, device: torch.device) -> _cached_signature_t:
        in_signature = (in_channels, in_volume)
        # TODO rever condição para atualização da cache
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
                        # print("here before")
                        activation_matrix_scalar, activation_tensor_scalar = activation.to_tensor(sequential_matrix)
                        # print("here after")
                        #print("ALPHA: ", sequential_matrix.values)
                        
                        tensors[layer] = (sequential_matrix.clone(), activation_matrix_scalar.detach().clone(), activation_tensor_scalar.detach().clone())

                    # Get ready for the next layer
                    in_channels, in_volume = out_channels, out_volume
                out_numel = out_channels * numpy.prod(out_volume)
                # Compute the backward cummulative product activation matrices
                backward_prod_matrices_extra = [None] * self.nlayers
                backward_prod_matrix_extra = torch.as_tensor(1, dtype=dtype, device=device)
                backward_prod_matrices_extra[self.nlayers - 1] = backward_prod_matrix_extra.detach().clone()
                for layer, (sequential_matrix, activation_matrix_scalar, _) in zip(range(self.nlayers - 2, -1, -1), tensors[-1:-self.nlayers:-1]):
                    backward_prod_matrix_extra = backward_prod_matrix_extra * activation_matrix_scalar
                    #print("BACK: ", backward_prod_matrix_extra===========================)
                    backward_prod_matrices_extra[layer] = backward_prod_matrix_extra.detach().clone()
                # Compute the final matrix and the final tensor encoding the Conformal Layers
                sequential_matrix, activation_matrix_scalar, _ = tensors[0]
                cached_matrix = IdentityMatrix(in_numel, dtype=dtype, device=device)
                # print("backward_prod_matrix_extra: ", backward_prod_matrix_extra)
                self._cached_matrix_extra = backward_prod_matrix_extra * activation_matrix_scalar
                # print("backward_prod_matrix_extra: ", backward_prod_matrix_extra)
                # print('cached_matrix_extra IN 134: ', self._cached_matrix_extra, '(', self._cached_matrix_extra.data_ptr(), ')')
                cached_tensor_extra = ZeroTensor((in_numel, in_numel), dtype=dtype, device=device)
                for backward_prod_matrix_extra, (sequential_matrix, _, activation_tensor_scalar) in zip(backward_prod_matrices_extra, tensors):
                    # print("backward_prod_matrix_extra: ", backward_prod_matrix_extra)
                    cached_matrix = torch.mm(sequential_matrix, cached_matrix)
                    if activation_tensor_scalar is not None:
                        # print("PEI")
                        # print("Activation_tensor_scalar_1: ", activation_tensor_scalar)
                        # print("HERE to current_extra")
                        current_extra = torch.mm(cached_matrix.t(), cached_matrix)
                        # print("HERE after current_extra")
                        # print("Activation_tensor_scalar_2: ", activation_tensor_scalar)
                        # print("PEI2")
                        # print('current_extra ANTES: ', current_extra.values)
                        current_extra = torch.mul(current_extra, activation_tensor_scalar)
                        # print("backward_prod_matrix_extra: ", backward_prod_matrix_extra)
                        # print('current_extra DEPOIS: ', current_extra.values)
                        cached_tensor_extra = torch.add(cached_tensor_extra, current_extra)
                        # print('cached_tensor_extra: ', cached_tensor_extra.values)
                self._cached_matrix = cached_matrix
                self._cached_tensor_extra = cached_tensor_extra
            else:
                self._cached_matrix = IdentityMatrix(in_numel, dtype=dtype, device=device)
                self._cached_matrix_extra = torch.as_tensor(1, dtype=dtype, device=device)
                self._cached_tensor_extra = ZeroTensor((in_numel, in_numel), dtype=dtype, device=device)
            # print('cached_matrix_extra IN 155: ', self._cached_matrix_extra, '(', self._cached_matrix_extra.data_ptr(), ')')
            # Set cached data as valid
            self._valid_cache = True
            self._cached_signature = (in_signature, (out_channels, out_volume), dtype)
            torch.cuda.synchronize()
        return self._cached_signature

    def forward(self, input: torch.Tensor):
        batches, in_channels, *in_volume = input.shape
        # If necessary, update cached data
        # print("UPDATING CACHE")
        _, (out_channels, out_volume), _ = self._update_cache(in_channels, tuple(in_volume), input.dtype, input.device)
        # print("DONE UPDATING CACHE")
        cached_matrix, cached_matrix_extra = self._cached_matrix.clone(), self._cached_matrix_extra.clone()
        # cached_matrix, cached_matrix_extra = self._cached_matrix, self._cached_matrix_extra
        # print("cached_matrix_extra OUT: ", cached_matrix_extra, '(', cached_matrix_extra.data_ptr(), ')')
        cached_tensor_extra = self._cached_tensor_extra.clone()
        # Reshape the input as a matrix where each batch entry corresponds to a column 
        input_as_matrix = input.view(batches, -1).t()
        input_as_matrix_extra = input_as_matrix.norm(dim=0, keepdim=True)
        # Apply the Conformal Layers
        # print("HERE to output_as_matrix")                    
        output_as_matrix = torch.mm(cached_matrix, input_as_matrix)
        output_as_matrix_extra = cached_matrix_extra * input_as_matrix_extra
        if not isinstance(cached_tensor_extra, ZeroTensor):
            # print("MM again")
            output_as_matrix_extra = output_as_matrix_extra + (torch.mm(cached_tensor_extra, input_as_matrix) * input_as_matrix).sum(dim=0, keepdim=True)
        output_as_matrix = output_as_matrix / output_as_matrix_extra
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
