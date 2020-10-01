from utils import unit_test
import numpy, torch


DIMENSIONS = [1, 2]
NATIVE_MODULES = [torch.nn.AvgPool1d, torch.nn.AvgPool2d]


BATCHES_START, BATCHES_END = 1, 3 + 1
IN_CHANNELS_START, IN_CHANNELS_END = 1, 3 + 1
IN_VOLUME_START, IN_VOLUME_END = 2, 5 + 1
KERNEL_SIZE_START = 2
STRIDE_START, STRIDE_END = 1, 4 + 1
PADDING_START = 0


def main():
    print('--- START AvgPool')
    case = 1
    for dimension, NativeModule in zip(DIMENSIONS, NATIVE_MODULES):
        for batches in range(BATCHES_START, BATCHES_END):
            for in_channels in range(IN_CHANNELS_START, IN_CHANNELS_END):
                for in_volume in numpy.ndindex(*numpy.full((dimension,), IN_VOLUME_END - IN_VOLUME_START, dtype=int)):
                    in_volume = numpy.add(in_volume, IN_VOLUME_START)
                    for kernel_size in numpy.ndindex(*(in_volume - KERNEL_SIZE_START + 1)):
                        kernel_size = numpy.add(kernel_size, KERNEL_SIZE_START)
                        for stride in numpy.ndindex(*numpy.full((dimension,), STRIDE_END - STRIDE_START, dtype=int)):
                            stride = numpy.add(stride, STRIDE_START)
                            for padding in numpy.ndindex(*(kernel_size - PADDING_START + 1)):
                                padding = numpy.add(padding, PADDING_START)
                                print(f'CASE #{case}: batches={batches}, in_channels={in_channels}, in_volume={*in_volume,}, kernel_size={*kernel_size,}, stride={*stride,}, padding={*padding,}')
                                unit_test(batches, in_channels, in_volume, NativeModule(kernel_size=tuple(kernel_size), stride=tuple(stride), padding=tuple(padding), ceil_mode=False, count_include_pad=True))
                                case += 1
    print('--- END AvgPool')


if __name__ == '__main__':
    main()
