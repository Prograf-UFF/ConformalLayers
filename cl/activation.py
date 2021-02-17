from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData
from .utils import ScalarTensor, SizeAny
from abc import abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple, Union
import math, numpy, torch
import os
import glob


class WrappedMinkowskiSRePro(torch.nn.Module):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedMinkowskiSRePro, self).__init__()
        self._owner = (owner,) # We use a tuple to avoid infinite recursion while PyTorch traverses the module's tree

    def forward(self, input: ForwardMinkowskiData) -> ForwardMinkowskiData:
        raise RuntimeError('Illegal call to the forward function. ConformalLayers was developed to evaluate the SRePro activation function differently in this module.')

    def to_tensor(self, alpha_upper: ScalarTensor) -> Tuple[ScalarTensor, ScalarTensor]:
        # Get the alpha parameter
        alpha = alpha_upper if self.owner.alpha is None else torch.as_tensor(self.owner.alpha, dtype=alpha_upper.dtype, device=alpha_upper.device)
        # Compute the last coefficient of the matrix
        matrix_scalar = alpha / 2
        # Compute the coefficient on the main diagonal of the last slice of the tensor
        tensor_scalar = 1 / (2 * alpha)
        # Return the scalars of the tensor representation of the activation function
        return matrix_scalar, tensor_scalar

    @property
    def owner(self) -> ConformalModule:
        return self._owner[0]


class WrappedTorchSRePro(torch.nn.Module):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedTorchSRePro, self).__init__()
        self._owner = (owner,) # We use a tuple to avoid infinite recursion while PyTorch traverses the module's tree

    def forward(self, input: ForwardTorchData) -> ForwardTorchData:
        (input, input_extra), alpha_upper = input
        batches, *in_dims = input.shape
        # Get the alpha parameter
        alpha = alpha_upper if self.owner.alpha is None else torch.as_tensor(self.owner.alpha, dtype=input.dtype, device=input.device)
        # Just for logging
        self._owner[0].log_alpha = alpha.detach().clone()
        # Apply the activation function
        flatted_input = input.view(batches, -1)
        input_sqr_norm = (flatted_input * flatted_input).sum(dim=1).unsqueeze(1)
        output_extra = (input_sqr_norm + input_extra * (alpha * alpha)) / (2 * alpha)
        alpha_upper = torch.as_tensor(1, dtype=alpha_upper.dtype, device=alpha_upper.device)

        # N_BATCHES_PER_EPOCH, N_ACTIVATION_LAYERS = 15, 3
        # path = os.path.join('Experiments', 'Tensors')
        # idx = len(glob.glob(os.path.join(path, '*.pth')))
        # EPOCH = idx // (N_BATCHES_PER_EPOCH * N_ACTIVATION_LAYERS)
        # LAYER = idx % N_ACTIVATION_LAYERS
        # BATCH = (idx // (N_ACTIVATION_LAYERS)) - (EPOCH * N_BATCHES_PER_EPOCH * N_ACTIVATION_LAYERS) + (EPOCH * 30)
        # print("\n\nEPOCH: ", EPOCH)
        # print("BATCH: ", BATCH)
        # print("LAYER: ", LAYER)
        # print('\n\n')
        # torch.save({
        #     'input' : input.detach().cpu(),
        #     'input_extra' : input_extra.detach().cpu(),
        #     'alpha' : alpha.detach().cpu(),
        #     'output_extra' : output_extra.detach().cpu(),
        # }, os.path.join(path, 'input_epoch_{}_batch_{}_layer_{}.pth'.format(EPOCH, BATCH, LAYER)))



        # Return the result
        return (input, output_extra), alpha_upper

    def to_tensor(self, alpha_upper: ScalarTensor) -> Tuple[ScalarTensor, ScalarTensor]:
        raise RuntimeError('Illegal call to the forward function. ConformalLayers was developed to evaluate the SRePro activation function differently in this module.')

    @property
    def owner(self) -> ConformalModule:
        return self._owner[0]


class BaseActivation(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(name=name)

    @abstractmethod
    def to_tensor(self, alpha_upper: ScalarTensor) -> Tuple[Optional[ScalarTensor], Optional[ScalarTensor]]:
        pass


class NoActivation(BaseActivation):
    def __init__(self) -> None:
        super(NoActivation, self).__init__()
        self._identity_module = torch.nn.Identity()

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        return input

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (*in_dims,)

    def to_tensor(self, alpha_upper: ScalarTensor) -> Tuple[None, None]:
        matrix_scalar = None
        tensor_scalar = None
        return matrix_scalar, tensor_scalar


class SRePro(BaseActivation):
    def __init__(self,
                 alpha: Optional[float]=None,
                 *, name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name=name)
        self._alpha = alpha
        self._torch_module = WrappedTorchSRePro(self)
        self._minkowski_module = WrappedMinkowskiSRePro(self)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['alpha'] = 'Automatic' if self._alpha is None else self._alpha
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        native_module = self._torch_module if self.training else self._minkowski_module
        return native_module(input)

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (*in_dims,)

    def to_tensor(self, alpha_upper: ScalarTensor) -> Tuple[ScalarTensor, ScalarTensor]:
        native_module = self._torch_module if self.training else self._minkowski_module
        return native_module.to_tensor(alpha_upper)
    
    @property
    def alpha(self) -> Optional[float]:
        return self._alpha
