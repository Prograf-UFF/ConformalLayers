from utils import unit_test
import numpy, torch


DIMENSIONS = [1, 2]
NATIVE_MODULES = [torch.nn.AvgPool1d, torch.nn.AvgPool2d]

BATCHES_START, BATCHES_END = 1, 3 + 1
IN_CHANNELS_START, IN_CHANNELS_END = 1, 3 + 1
IN_VOLUME_START, IN_VOLUME_END = 2, 5 + 1
STRIDE_START, STRIDE_END = 1, 4 + 1


def main():
    print('--- START AvgPool')
    sum_native_time = 0
    sum_cl_time = 0
    sum_cl_cached_time = 0
    case = 1
    for dimension, NativeModule in zip(DIMENSIONS, NATIVE_MODULES):
        for batches in range(BATCHES_START, BATCHES_END):
            for in_channels in range(IN_CHANNELS_START, IN_CHANNELS_END):
                for in_volume in numpy.ndindex(*numpy.full((dimension,), IN_VOLUME_END - IN_VOLUME_START, dtype=int)):
                    in_volume = numpy.add(in_volume, IN_VOLUME_START)
                    for kernel_size in numpy.ndindex(*(in_volume - 1)):
                        kernel_size = numpy.add(kernel_size, 2)
                        for stride in numpy.ndindex(*numpy.full((dimension,), STRIDE_END - STRIDE_START, dtype=int)):
                            stride = numpy.add(stride, STRIDE_START)
                            for padding in numpy.ndindex(*(kernel_size // 2 + 1)):
                                print(f'CASE #{case}: batches={batches}, in_channels={in_channels}, in_volume={*in_volume,}, kernel_size={*kernel_size,}, stride={*stride,}, padding={*padding,}')
                                native_time, cl_time, cl_cached_time = unit_test(batches, in_channels, in_volume, NativeModule(kernel_size=tuple(kernel_size), stride=tuple(stride), padding=tuple(padding), ceil_mode=False, count_include_pad=True))
                                sum_native_time += native_time
                                sum_cl_time += cl_time
                                sum_cl_cached_time += cl_cached_time
                                case += 1
    print(f'--- Native: {sum_native_time / (case - 1)} sec; CL: {sum_cl_time / (case - 1)} sec; Cached CL: {sum_cl_cached_time / (case - 1)} sec')
    print('--- END AvgPool')


if __name__ == '__main__':
    main()
