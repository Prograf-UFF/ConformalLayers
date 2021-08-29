from utils import unit_test
import numpy
import torch


DIMENSIONS = [1, 2, 3]

BATCHES_START, BATCHES_END = 1, 3 + 1
IN_CHANNELS_START, IN_CHANNELS_END = 1, 3 + 1
IN_VOLUME_START, IN_VOLUME_END = 2, 5 + 1


def main():
    print('--- START Flatten')
    times_sum = torch.zeros(5, dtype=torch.float32, device='cpu')
    case = 1
    NativeModule = torch.nn.Flatten
    for dimension in DIMENSIONS:
        for batches in range(BATCHES_START, BATCHES_END):
            for in_channels in range(IN_CHANNELS_START, IN_CHANNELS_END):
                for in_volume in numpy.ndindex(*numpy.full((dimension,), IN_VOLUME_END - IN_VOLUME_START, dtype=int)):
                    in_volume = numpy.add(in_volume, IN_VOLUME_START)
                    print(f'CASE #{case}: batches={batches}, in_channels={in_channels}, in_volume={*in_volume,}')
                    times_sum += unit_test(batches, (in_channels, *in_volume), NativeModule())
                    case += 1
    print(f'  - Train')
    print(f'    Native: {times_sum[0] / (case - 1): 1.8f} sec; \tCL: {times_sum[1] / (case - 1): 1.8f} sec')
    print(f'  - Eval 1')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[3] / (case - 1): 1.8f} sec')
    print(f'  - Eval 2')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[4] / (case - 1): 1.8f} sec')
    print('--- END Flatten')


if __name__ == '__main__':
    main()
