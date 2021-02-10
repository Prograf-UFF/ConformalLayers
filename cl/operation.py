from .convolution import Conv1d
from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData
from .utils import DenseTensor, ScalarTensor, SizeAny
from collections import OrderedDict
from typing import Optional, Tuple, Union
import MinkowskiEngine as me
import numpy, torch


class WrappedMinkowskiLinear(torch.nn.Module):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedMinkowskiLinear, self).__init__()
        self._owner = (owner,) # We use a tuple to avoid infinite recursion while PyTorch traverses the module's tree
        self._function = me.MinkowskiConvolutionFunction()
        kernel_generator = me.KernelGenerator(kernel_size=(1,), stride=1, dilation=(1,), dimension=1)
        self._kernel_size = torch.as_tensor((1,), dtype=torch.int32, device='cpu')
        self._kernel_region_type, self._kernel_region_offset, kernel_volume = kernel_generator.get_kernel((1,), False)
        self._kernel = torch.empty((kernel_volume, owner.in_features, owner.out_features), dtype=owner.weight.dtype, device=owner.weight.device)

    def forward(self, input: ForwardMinkowskiData) -> ForwardMinkowskiData:
        input, alpha_upper = input
        in_coords = input.coords
        indices_per_batch = input.decomposed_coordinates
        # Compute the complete set of coordinates for evaluating the module
        index_end = in_coords[:, 1:].max(0)[0] + 1 #TODO How to replace the max function call by some predefined value?
        out_coords = torch.cat(tuple(torch.stack(torch.meshgrid(torch.as_tensor((batch,), dtype=torch.int32, device=in_coords.device), *map(lambda start, end: torch.arange(int(start), int(end), dtype=torch.int32, device=in_coords.device),
            indices.min(0)[0],
            torch.min(index_end, (indices.max(0)[0] + 1)))), dim=-1).view(-1, 1 + input.dimension) for batch, indices in enumerate(indices_per_batch)), dim=0)
        #TODO assert (torch.abs(output.feats) <= 1e-6).all(), 'Os limites do arange(...) precisam ser ajustados, pois coordenadas irrelevantes são geradas em casos a serem investigados
        # Evaluate the module
        self._kernel = self._kernel.to(self.owner.weight.device)
        self._kernel.copy_(self.owner.weight.T.reshape(*self._kernel.shape)) # We don't know why Minkowski Engine convolution does not work with the view
        out_coords_key = input.coords_man.create_coords_key(out_coords, tensor_stride=1, force_creation=True, force_remap=True, allow_duplicate_coords=True)
        out_feats = self._function.apply(input.feats, self._kernel, input.tensor_stride, 1, self._kernel_size, (1,), self._kernel_region_type, self._kernel_region_offset, input.coords_key, out_coords_key, input.coords_man)
        alpha_upper = alpha_upper * torch.linalg.norm(self.owner.weight, ord=1) # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
        # Map the first indices to zeros and compress the resulting coordinates when needed
        output = me.SparseTensor(out_feats, coords_key=out_coords_key, coords_manager=input.coords_man)
        return output, alpha_upper

    @property
    def owner(self) -> ConformalModule:
        return self._owner[0]


class Linear(ConformalModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 *, name: Optional[str]=None) -> None:
        super(Linear, self).__init__(name=name)
        self._torch_module = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self._minkowski_module = WrappedMinkowskiLinear(self)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['in_features'] = self.in_features
        entries['out_features'] = self.out_features
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        if self.training:
            (input, input_extra), alpha_upper = input
            if input.dim() != 2:
                raise NotImplementedError() # TODO implement the general case
            output = self._torch_module(input)
            alpha_upper = alpha_upper * torch.linalg.norm(self._torch_module.weight, ord=1) # Apply the Young's convolution inequality with p = 2, q = 1, and r = 2 (https://en.m.wikipedia.org/wiki/Young%27s_convolution_inequality).
            return (output, input_extra), alpha_upper
        else:
            return self._minkowski_module(input)

    def output_dims(self, *in_size: int) -> SizeAny:
        return (*in_size[:-1], self.out_features)

    @property
    def in_features(self) -> int:
        return self._torch_module.in_features

    @property
    def out_features(self) -> int:
        return self._torch_module.out_features

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._torch_module.weight
