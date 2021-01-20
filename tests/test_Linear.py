from utils import unit_test
import numpy, torch


DIMENSIONS = [2, 3]

BATCHES_START, BATCHES_END = 1, 3 + 1
DIMS_START, DIMS_END = 2, 5 + 1


def main():
    print('--- START Linear')
    times_sum = torch.zeros(5, dtype=torch.float32, device='cpu')
    case = 1
    NativeModule = torch.nn.Linear
    for dimension in DIMENSIONS:
        for batches in range(BATCHES_START, BATCHES_END):
            for other_dims in numpy.ndindex(*numpy.full((dimension-1,), DIMS_END - DIMS_START, dtype=int)):
                other_dims = numpy.add(other_dims, DIMS_START)
                for in_features in range(DIMS_START, DIMS_END):
                    for out_features in range(DIMS_START, DIMS_END):
                        print(f'CASE #{case}: batches={batches}, other_dims={*other_dims,}, in_features={in_features}, out_features={out_features}')
                        times_sum += unit_test(batches, (*other_dims, in_features), NativeModule(in_features=in_features, out_features=out_features, bias=False))
                        case += 1
    print(f'  - Train')
    print(f'    Native: {times_sum[0] / (case - 1): 1.8f} sec; \tCL: {times_sum[1] / (case - 1): 1.8f} sec')
    print(f'  - Eval 1')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[3] / (case - 1): 1.8f} sec')
    print(f'  - Eval 2')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[4] / (case - 1): 1.8f} sec')
    print('--- END AvgPool')


if __name__ == '__main__':
    main()
