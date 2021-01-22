from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData
from .utils import SizeAny, ravel_multi_index, unravel_index
from collections import OrderedDict
from typing import Optional, Union
import MinkowskiEngine as me
import numpy, torch


class Flatten(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(Flatten, self).__init__(name=name)
        self._torch_module = torch.nn.Flatten(
            start_dim=1, #TODO Implement start_dim != 1
            end_dim=-1)  #TODO Implement end_dim != -1

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['start_dim'] = self.start_dim
        entries['end_dim'] = self.end_dim
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        # Evaluate using PyTorch
        if self.training:
            (input, input_extra), alpha_upper = input
            output = self._torch_module(input)
            batches, *out_dims = output.shape
            output_extra = input_extra.view(batches, *map(lambda _: 1, range(len(out_dims))))
            return (output, output_extra), alpha_upper
        # Evaluate using MinkowskiEngine
        else:
            input, alpha_upper = input
            # Compute the shape of the input tensor
            dense_dim = input.feats.shape[1]
            sparse_dims = input.coords[:, 1:].max(0)[0] + 1 #TODO How to replace the max function call by some predefined value?
            # Compute the coordinates of the input entries
            in_coords = input.coords.view(-1, 1 + len(sparse_dims), 1).expand(-1, -1, dense_dim).permute(0, 2, 1)
            in_coords = torch.cat((in_coords, torch.empty((len(in_coords), dense_dim, 1), dtype=torch.int32, device=in_coords.device)), 2)
            for ind in range(dense_dim):
                in_coords[:, ind, -1] = ind
            in_coords = in_coords.view(-1, len(sparse_dims) + 2)
            in_numel = dense_dim * int(sparse_dims.prod())
            # Flatten the input features and their coordinates
            out_feats = input.feats.view(-1, 1)
            out_coords = torch.stack(unravel_index(ravel_multi_index(tuple(in_coords[:, dim] for dim in (0, -1, *range(1, in_coords.shape[1] - 1))), (in_numel, dense_dim, *sparse_dims)), (in_numel, in_numel))).t()  
            return me.SparseTensor(out_feats, out_coords), alpha_upper

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (numpy.prod(in_dims),)

    @property
    def start_dim(self):
        return self._torch_module.start_dim

    @property
    def end_dim(self):
        return self._torch_module.end_dim
