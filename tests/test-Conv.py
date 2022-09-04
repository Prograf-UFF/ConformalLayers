from utils import unit_test
import numpy
import torch


DIMENSIONS = [1, 2]
NATIVE_MODULES = [torch.nn.Conv1d, torch.nn.Conv2d]

BATCHES_START, BATCHES_END = 1, 3 + 1
GROUPS_START, GROUPS_END = 1, 3 + 1
IN_CHANNELS_START, IN_CHANNELS_END = 1, 3 + 1  # Those values will be multiplied by `groups`.
OUT_CHANNELS_START, OUT_CHANNELS_END = 1, 3 + 1  # Those values will be multiplied by `groups`.
IN_VOLUME_START, IN_VOLUME_END = 2, 5 + 1
STRIDE_START, STRIDE_END = 1, 4 + 1


def main():
    print('--- START Conv')
    times_sum = torch.zeros(5, dtype=torch.float32, device='cpu')
    case = 1
    for dimension, NativeModule in zip(DIMENSIONS, NATIVE_MODULES):
        for batches in range(BATCHES_START, BATCHES_END):
            for groups in range(GROUPS_START, GROUPS_END):
                for in_channels in range(IN_CHANNELS_START * groups, IN_CHANNELS_END * groups):
                    for out_channels in range(OUT_CHANNELS_START * groups, OUT_CHANNELS_END * groups):
                        for in_volume in numpy.ndindex(*numpy.full((dimension,), IN_VOLUME_END - IN_VOLUME_START, dtype=int)):
                            in_volume = numpy.add(in_volume, IN_VOLUME_START)
                            for kernel_size in numpy.ndindex(*(in_volume - 1)):
                                kernel_size = numpy.add(kernel_size, 2)
                                for stride in numpy.ndindex(*numpy.full((dimension,), STRIDE_END - STRIDE_START, dtype=int)):
                                    stride = numpy.add(stride, STRIDE_START)
                                    for padding in numpy.ndindex(*(kernel_size + 1)):
                                        for dilation in numpy.ndindex(*(numpy.maximum((in_volume - 1) // (kernel_size - 1), 1) - 1)):
                                            dilation = numpy.add(dilation, 1)
                                            print(f'CASE #{case}: batches={batches}, in_channels={in_channels}, out_channels={out_channels}, in_volume={*in_volume,}, kernel_size={*kernel_size,}, stride={*stride,}, padding={*padding,}, dilation={*dilation,}, groups={groups}')
                                            module = NativeModule(in_channels=in_channels, out_channels=out_channels, kernel_size=tuple(kernel_size), stride=tuple(stride), padding=tuple(padding), dilation=tuple(dilation), groups=groups, bias=False, padding_mode='zeros')
                                            times_sum += unit_test(batches, (in_channels, *in_volume), module)
                                            case += 1
    print(f'  - Train')
    print(f'    Native: {times_sum[0] / (case - 1): 1.8f} sec; \tCL: {times_sum[1] / (case - 1): 1.8f} sec')
    print(f'  - Eval 1')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[3] / (case - 1): 1.8f} sec')
    print(f'  - Eval 2')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[4] / (case - 1): 1.8f} sec')
    print('--- END Conv')


if __name__ == '__main__':
    main()
