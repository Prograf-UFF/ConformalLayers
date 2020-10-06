try:
    import cl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import cl

import time, torch

class NativeNet(object):
    def __init__(self, in_volume, *native_modules):
        self.modules = torch.nn.Sequential(*native_modules)

    def __call__(self, input):
        return self.modules(input)


class CLNet(object):
    def __init__(self, in_volume, *native_modules):
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
                self._copy_kernel(module.weight, modules[-1].kernel)
            elif isinstance(module, torch.nn.Conv2d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel)
            elif isinstance(module, torch.nn.Conv3d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.Conv3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel)
            elif isinstance(module, torch.nn.ConvTranspose1d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.ConvTranspose1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel)
            elif isinstance(module, torch.nn.ConvTranspose2d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.ConvTranspose2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel)
            elif isinstance(module, torch.nn.ConvTranspose3d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.ConvTranspose3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                self._copy_kernel(module.weight, modules[-1].kernel)
            else:
                raise NotImplementedError()
        self.modules = cl.ConformalLayers(*modules)

    def __call__(self, input):
        return self.modules(input)

    def _copy_kernel(self, src, dst):
        dst.data.copy_(src.data.T.reshape(*dst.data.shape))


def unit_test(batches, in_channels, in_volume, *native_modules):
    tol = 1e-6
    # Bind native net and conformal layer-based net
    native_net = NativeNet(in_volume, *native_modules)
    cl_net = CLNet(in_volume, *native_modules)
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
