from utils import unit_test
import numpy, torch


DIMENSIONS = [2, 3]

BATCHES_START, BATCHES_END = 1, 3 + 1
DIMS_START, DIMS_END = 2, 5 + 1


def main():
    print('--- START Linear')
    sum_native_time = 0
    sum_cl_time = 0
    sum_cl_cached_time = 0
    case = 1
    NativeModule = torch.nn.Linear
    for dimension in DIMENSIONS:
        for batches in range(BATCHES_START, BATCHES_END):
            for other_dims in numpy.ndindex(*numpy.full((dimension-1,), DIMS_END - DIMS_START, dtype=int)):
                other_dims = numpy.add(other_dims, DIMS_START)
                for in_features in range(DIMS_START, DIMS_END):
                    for out_features in range(DIMS_START, DIMS_END):
                        print(f'CASE #{case}: batches={batches}, other_dims={*other_dims,}, in_features={in_features}, out_features={out_features}')
                        native_time, cl_time, cl_cached_time = unit_test(batches, (*other_dims, in_features), NativeModule(in_features=in_features, out_features=out_features, bias=False))
                        sum_native_time += native_time
                        sum_cl_time += cl_time
                        sum_cl_cached_time += cl_cached_time
                        case += 1
    print(f'--- Native: {sum_native_time / (case - 1)} sec; CL: {sum_cl_time / (case - 1)} sec; Cached CL: {sum_cl_cached_time / (case - 1)} sec')
    print('--- END AvgPool')


if __name__ == '__main__':
    main()
