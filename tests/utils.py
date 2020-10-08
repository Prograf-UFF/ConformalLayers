try:
    import cl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import cl

from typing import Tuple
import time, torch

class NativeNet(object):
    def __init__(self, *native_modules: torch.nn.Module):
        self.modules = torch.nn.Sequential(*native_modules)

    def __call__(self, input):
        return self.modules(input)


class CLNet(object):
    def __init__(self, *native_modules: torch.nn.Module):
        modules = list()
        for module in native_modules:
            if isinstance(module, torch.nn.AvgPool1d):
                assert not module.ceil_mode and module.count_include_pad
                modules.append(cl.AvgPool1d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding))
            elif isinstance(module, torch.nn.AvgPool2d):
                assert not module.ceil_mode and module.count_include_pad and not module.divisor_override
                modules.append(cl.AvgPool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding))
            elif isinstance(module, torch.nn.AvgPool3d):
                assert not module.ceil_mode and module.count_include_pad and not module.divisor_override
                modules.append(cl.AvgPool3d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding))
            elif isinstance(module, torch.nn.Conv1d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel, False)
            elif isinstance(module, torch.nn.Conv2d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel, False)
            elif isinstance(module, torch.nn.Conv3d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.Conv3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel, False)
            elif isinstance(module, torch.nn.ConvTranspose1d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.ConvTranspose1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel, True)
            elif isinstance(module, torch.nn.ConvTranspose2d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.ConvTranspose2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel, True)
            elif isinstance(module, torch.nn.ConvTranspose3d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.ConvTranspose3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel, True)
            else:
                raise NotImplementedError()
        self.modules = cl.ConformalLayers(*modules)

    def __call__(self, input: torch.Tensor):
        return self.modules(input)

    def _copy_kernel(self, src: torch.nn.Parameter, dst: torch.nn.Parameter, transposed: bool):
        if transposed:
            dst.data.copy_(src.data.permute(1, 0, *range(2, src.data.dim())).T.reshape(*dst.data.shape))
        else:
            dst.data.copy_(src.data.T.reshape(*dst.data.shape))


def unit_test(batches: int, in_channels: int, in_volume: Tuple[int, ...], *native_modules: torch.nn.Module):
    tol = 1e-6
    # Bind native net and conformal layer-based net
    native_net = NativeNet(*native_modules)
    cl_net = CLNet(*native_modules)
    # Create input data
    input = torch.rand(batches, in_channels, *in_volume)
    # Compute resulting data
    start_time = time.time()
    y_native = native_net(input)
    native_time = time.time() - start_time
    start_time = time.time()
    y_cl = cl_net(input)
    cl_time = time.time() - start_time
    start_time = time.time()
    y_cl = cl_net(input)
    cl_cached_time = time.time() - start_time
    # Compare results
    if torch.max(torch.abs(y_native - y_cl)) > tol:
        raise RuntimeError(f'\ny_native = {y_native}\ny_cl = {y_cl}')
    return native_time, cl_time, cl_cached_time
