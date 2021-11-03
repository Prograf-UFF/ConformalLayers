from typing import Tuple
import cl
import time, warnings
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    warnings.warn(f'The device was set to {DEVICE}.', RuntimeWarning)


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
                modules[-1].weight.data.copy_(module.weight.data)
            elif isinstance(module, torch.nn.Conv2d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                modules[-1].weight.data.copy_(module.weight.data)
            elif isinstance(module, torch.nn.Conv3d):
                assert module.groups == 1 and not module.bias and module.padding_mode == 'zeros'
                modules.append(cl.Conv3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation))
                modules[-1].weight.data.copy_(module.weight.data)
            elif isinstance(module, torch.nn.Flatten):
                assert module.start_dim == 1 and module.end_dim == -1
                modules.append(cl.Flatten())
            else:
                raise NotImplementedError()
        self.modules = cl.ConformalLayers(*modules, pruning_threshold=None)

    def __call__(self, input: torch.Tensor):
        return self.modules(input)


def unit_test(batches: int, in_dims: Tuple[int, ...], *native_modules: torch.nn.Module):
    #TODO tol = 1e-6
    tol = 1e-3
    # Bind native net and Conformal Layer-based net
    native_net = NativeNet(*native_modules)
    native_net.modules.to(DEVICE)
    cl_net = CLNet(*native_modules)
    cl_net.modules.to(DEVICE)
    # Create input data
    input = torch.rand(batches, *in_dims).to(DEVICE)
    unit_input = input / torch.linalg.norm(input.view(batches, -1), ord=2, dim=1).view(batches, *map(lambda _: 1, range(len(in_dims))))
    # Compute resulting data
    native_net.modules.train()
    start_time = time.time()
    output_native_train = native_net(unit_input)
    native_train_time = time.time() - start_time
    #
    cl_net.modules.train()
    start_time = time.time()
    output_cl_train = cl_net(input)
    cl_train_time = time.time() - start_time
    #
    native_net.modules.eval()
    start_time = time.time()
    output_native_eval = native_net(unit_input)
    native_eval_time = time.time() - start_time
    #
    cl_net.modules.eval()
    start_time = time.time()
    output_cl_eval1 = cl_net(input)
    cl_eval1_time = time.time() - start_time
    #
    cl_net.modules.eval()
    start_time = time.time()
    output_cl_eval2 = cl_net(input)
    cl_eval2_time = time.time() - start_time
    # Compare results
    if torch.max(torch.abs(output_native_train - output_cl_train)) > tol:
        raise RuntimeError(f'\nTrain\nnative = {output_native_train}\ncl = {output_cl_train}')
    if torch.max(torch.abs(output_native_eval - output_cl_eval1)) > tol:
        raise RuntimeError(f'\nEval 1\nnative = {output_native_eval}\ncl = {output_cl_eval1}')
    if torch.max(torch.abs(output_native_eval - output_cl_eval2)) > tol:
        raise RuntimeError(f'\nEval 2\nnative = {output_native_eval}\ncl = {output_cl_eval2}')
    # Return elapsed times
    return torch.as_tensor((native_train_time, cl_train_time, native_eval_time, cl_eval1_time, cl_eval2_time), dtype=torch.float32, device='cpu')
