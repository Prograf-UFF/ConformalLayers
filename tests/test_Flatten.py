from utils import unit_test
import numpy, torch


DIMENSIONS = [1, 2, 3]

BATCHES_START, BATCHES_END = 1, 3 + 1
IN_CHANNELS_START, IN_CHANNELS_END = 1, 3 + 1
IN_VOLUME_START, IN_VOLUME_END = 2, 5 + 1


def main():
    print('--- START Flatten')
    sum_native_time = 0
    sum_cl_time = 0
    sum_cl_cached_time = 0
    case = 1
    NativeModule = torch.nn.Flatten
    for dimension in DIMENSIONS:
        for batches in range(BATCHES_START, BATCHES_END):
            for in_channels in range(IN_CHANNELS_START, IN_CHANNELS_END):
                for in_volume in numpy.ndindex(*numpy.full((dimension,), IN_VOLUME_END - IN_VOLUME_START, dtype=int)):
                    in_volume = numpy.add(in_volume, IN_VOLUME_START)
                    print(f'CASE #{case}: batches={batches}, in_channels={in_channels}, in_volume={*in_volume,}')
                    native_time, cl_time, cl_cached_time = unit_test(batches, (in_channels, *in_volume), NativeModule())
                    sum_native_time += native_time
                    sum_cl_time += cl_time
                    sum_cl_cached_time += cl_cached_time
                    case += 1
    print(f'--- Native: {sum_native_time / (case - 1)} sec; CL: {sum_cl_time / (case - 1)} sec; Cached CL: {sum_cl_cached_time / (case - 1)} sec')
    print('--- END Flatten')


if __name__ == '__main__':
    main()
