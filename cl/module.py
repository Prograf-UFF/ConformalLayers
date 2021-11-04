from .utils import DenseTensor, ScalarTensor, SizeAny
from abc import abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple, Union
import MinkowskiEngine as me
import torch


# (X, alpha_upper), where X is the canonical representation of the input including all coordinates,
# and alpha_upper is the upper limit for hypersphere radius
ForwardMinkowskiData = Tuple[me.SparseTensor, ScalarTensor]

# ((Xe, Xw), alpha_upper), where Xe includes the Euclidean coordinates of the input,
# Xw is the homogeneous coordinate of the input, and alpha_upper is the upper limit for hypersphere radius
ForwardTorchData = Tuple[Tuple[DenseTensor, ScalarTensor], ScalarTensor]


class ConformalModule(torch.nn.Module):

    def __init__(self, *, name: Optional[str] = None) -> None:
        super(ConformalModule, self).__init__()
        self._name = name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({", ".join(map(lambda pair: "{}={}".format(*pair), self._repr_dict().items()))})'

    def _repr_dict(self) -> OrderedDict:
        entries = OrderedDict()
        if self._name is not None:
            entries['name'] = self.name
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        raise NotImplementedError  # This method must be implemented by the subclasses

    @abstractmethod
    def output_dims(self, *in_size: int) -> SizeAny:
        pass

    @property
    def name(self) -> Optional[str]:
        return self._name


class WrappedMinkowskiStridedOperation(torch.nn.Module):
    
    def __init__(self, owner: ConformalModule) -> None:
        super(WrappedMinkowskiStridedOperation, self).__init__()
        # Declare basic properties
        self._owner = (owner,)  # We use a tuple to avoid infinite recursion while PyTorch traverses the module's tree
        # Compute some constant values and keep them
        kernel_origin = owner.dilation * torch.div(owner.kernel_size - 1, 2, rounding_mode='floor') * (owner.kernel_size % 2)
        dilated_kernel_size = owner.dilation * (owner.kernel_size - 1) + 1
        self._kernel_start_offset = kernel_origin - dilated_kernel_size + 1
        self._kernel_end_offset = kernel_origin
        self._index_start_offset = kernel_origin - owner.padding
        self._index_end_offset = kernel_origin + owner.padding - dilated_kernel_size + 2

    @abstractmethod
    def _apply_function(self, input: me.SparseTensor, alpha_upper: ScalarTensor, out_coordinate_map_key: me.CoordinateMapKey) -> Tuple[DenseTensor, ScalarTensor]:
        pass

    def forward(self, input: ForwardMinkowskiData) -> ForwardMinkowskiData:
        input, alpha_upper = input
        in_coords = input.coordinates
        indices_per_batch = input.decomposed_coordinates
        # Compute the complete set of coordinates for evaluating the module
        index_end_offset = self._index_end_offset.to(in_coords.device)
        index_start = self._index_start_offset.to(in_coords.device)
        index_end = in_coords[:, 1:].max(0)[0] + index_end_offset
        kernel_start_offset = self._kernel_start_offset.to(in_coords.device)
        kernel_end_offset = self._kernel_end_offset.to(in_coords.device)
        stride = self.owner.stride.to(in_coords.device)
        out_coords = torch.cat(tuple(
            torch.stack(torch.meshgrid(torch.as_tensor((batch,), dtype=torch.int32, device=in_coords.device),
                                       *map(lambda start, end, step: torch.arange(int(start), int(end), int(step), dtype=torch.int32, device=in_coords.device),
                                            torch.max(index_start, (torch.div(indices.min(0)[0] + kernel_start_offset - index_start, stride, rounding_mode='floor')) * stride + index_start),
                                            torch.min(index_end, (torch.div(indices.max(0)[0] + kernel_end_offset - index_start, stride, rounding_mode='floor') + 1) * stride + index_start),
                                            stride)), dim=-1).view(-1, 1 + input.dimension)
            for batch, indices in enumerate(indices_per_batch)), dim=0)
        # Evaluate the module
        out_coordinate_map_key = me.CoordinateMapKey([1 for _ in range(out_coords.size(1) - 1)], '')
        out_coordinate_map_key, _ = input._manager.insert_and_map(out_coords, *out_coordinate_map_key.get_key())
        out_feats, alpha_upper = self._apply_function(input, alpha_upper, out_coordinate_map_key)
        # Map the first indices to zeros and compress the resulting coordinates when needed
        if (index_start != 0).any():
            out_coords[:, 1:] -= index_start
            if (self.owner.stride != 1).any():
                out_coords[:, 1:] = torch.div(out_coords[:, 1:], stride, rounding_mode='floor')
            output = me.SparseTensor(out_feats, out_coords, coordinate_manager=input.coordinate_manager)
        elif (self.owner.stride != 1).any():
            out_coords[:, 1:] = torch.div(out_coords[:, 1:], stride, rounding_mode='floor')
            output = me.SparseTensor(out_feats, out_coords, coordinate_manager=input.coordinate_manager)
        else:
            output = me.SparseTensor(out_feats, coordinate_map_key=out_coordinate_map_key, coordinate_manager=input.coordinate_manager)
        return output, alpha_upper

    @property
    def owner(self) -> ConformalModule:
        return self._owner[0]
