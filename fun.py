import torch
import cl.torch as cl

class DenseNet(object):
    def __init__(self, in_volume, *dense_modules):
        self.modules = torch.nn.Sequential(*dense_modules)

    def __call__(self, input):
        return self.modules(input)


class SparseNet(object):
    def __init__(self, in_volume, *dense_modules):
        modules = list()
        for dense_module in dense_modules:
            if isinstance(dense_module, torch.nn.Conv1d):
                modules.append(cl.Conv1d(
                    in_channels=dense_module.in_channels,
                    out_channels=dense_module.out_channels,
                    kernel_size=dense_module.kernel_size,
                    stride=dense_module.stride,
                    padding=dense_module.padding,
                    dilation=dense_module.dilation))
                self._copy_kernel(dense_module.weight, modules[-1].kernel)
            elif isinstance(dense_module, torch.nn.Conv2d):
                modules.append(cl.Conv2d(
                    in_channels=dense_module.in_channels,
                    out_channels=dense_module.out_channels,
                    kernel_size=dense_module.kernel_size,
                    stride=dense_module.stride,
                    padding=dense_module.padding,
                    dilation=dense_module.dilation))
                self._copy_kernel(dense_module.weight, modules[-1].kernel)
            elif isinstance(dense_module, torch.nn.Conv3d):
                modules.append(cl.Conv3d(
                    in_channels=dense_module.in_channels,
                    out_channels=dense_module.out_channels,
                    kernel_size=dense_module.kernel_size,
                    stride=dense_module.stride,
                    padding=dense_module.padding,
                    dilation=dense_module.dilation))
                self._copy_kernel(dense_module.weight, modules[-1].kernel)
            elif isinstance(dense_module, torch.nn.AvgPool1d):
                modules.append(cl.AvgPool1d(
                    kernel_size=dense_module.kernel_size,
                    stride=dense_module.stride,
                    dilation=dense_module.dilation))
            elif isinstance(dense_module, torch.nn.AvgPool2d):
                modules.append(cl.AvgPool2d(
                    kernel_size=dense_module.kernel_size,
                    stride=dense_module.stride,
                    dilation=dense_module.dilation))
            elif isinstance(dense_module, torch.nn.AvgPool3d):
                modules.append(cl.AvgPool3d(
                    kernel_size=dense_module.kernel_size,
                    stride=dense_module.stride,
                    dilation=dense_module.dilation))
            else:
                raise NotImplementedError()
        self.modules = cl.ConformalLayers(*modules)

    def __call__(self, input):
        return self.modules(input)

    def _copy_kernel(self, dense_src, sparse_dst):
        sparse_dst.data.copy_(dense_src.data.T.reshape(*sparse_dst.data.shape))


def disp(tensor, end='\n', float_mask='', name=None):
    dims = len(tensor.shape)
    if name is not None:
        print(f'{name} = ', end='')
    if dims == 0:
        print('empty', end=end)
    elif dims == 1:
        mask = '{0:' + float_mask + '}'
        print('{', end='')
        print(*(mask.format(tensor[ind]) for ind in range(tensor.shape[0])), sep=', ', end='')
        print('}', end=end)
    else:
        print('{', end='')
        if tensor.shape[0] > 0:
            disp(tensor[0, ...], end='', float_mask=float_mask)
            for ind in range(1, tensor.shape[0]):
                print(', ', end='')
                disp(tensor[ind, ...], end='', float_mask=float_mask)
        print('}', end=end)


def unit_test(batches, in_volume, *dense_modules):
    tol = 1e-6
    in_channels = dense_modules[0].in_channels
    # Módulos que correspondem ao produtório de U's
    dense_net = DenseNet(in_volume, *dense_modules)
    sparse_net = SparseNet(in_volume, *dense_modules)
    # Gerar dados de entrada para teste
    input = torch.rand(batches, in_channels, *in_volume)
    # Comparar o que se obtém processando uma entada qualquer utilizando as implementações nativas
    y_dense = dense_net(input)
    y_sparse = sparse_net(input)
    assert torch.max(torch.abs(y_dense - y_sparse)) <= tol, f'\ny_dense = {y_dense}\ny_sparse = {y_sparse}'


def main():
    print('--- START')
    case = 1
    for batches in range(1, 4):
        for input_size in range(3, 10):
            for in_channels in range(1, 4):
                for out_channels in range(1, 4):
                    for kernel_size in range(2, input_size + 1):
                        for stride in range(1, 5):
                            for padding in range(0, kernel_size + 1):
                                for dilation in range(1, max((input_size - 1) // (kernel_size - 1), 1) + 1):
                                    print(f'CASE #{case}: batches = {batches}, input_size = {input_size}, in_channels = {in_channels}, out_channels = {out_channels}, kernel_size = {kernel_size}, stride = {stride}, padding = {padding}, dilation = {dilation}')
                                    unit_test(batches, (input_size,), torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,), stride=stride, padding=padding, dilation=dilation, groups=1, bias=False, padding_mode='zeros'))
                                    case += 1
    print('--- END')


if __name__ == '__main__':
    main()
