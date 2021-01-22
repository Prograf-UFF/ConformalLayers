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
    error = list() #TODO Parei aqui!
    for dimension in DIMENSIONS:
        for batches in range(BATCHES_START, BATCHES_END):
            for other_dims in numpy.ndindex(*numpy.full((dimension-1,), DIMS_END - DIMS_START, dtype=int)):
                other_dims = numpy.add(other_dims, DIMS_START)
                for in_features in range(DIMS_START, DIMS_END):
                    for out_features in range(DIMS_START, DIMS_END):
                        #TODO Parei aqui!
                        input = torch.rand(batches, *other_dims, in_features)
                        batches, *in_dims = input.shape

                        linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
                        output = linear(input)
                        
                        weight = linear.weight
                        
                        module = torch.nn.Conv1d(in_features, out_features, (1,), bias=False)
                        module.weight.data.copy_(weight.view(out_features, in_features, 1))

                        input_ = input.view(batches, -1, in_features).transpose(1, 2)
                        output_ = module(input_).transpose(1, 2).view(batches, *in_dims[:-1], out_features)

                        error.append(float((output-output_).abs().max()))
                        
                        print(f'CASE #{case}: batches={batches}, other_dims={*other_dims,}, in_features={in_features}, out_features={out_features}')
                        times_sum += unit_test(batches, (*other_dims, in_features), NativeModule(in_features=in_features, out_features=out_features, bias=False))

                        case += 1
    print(torch.as_tensor(error).max())
    
    print(f'  - Train')
    print(f'    Native: {times_sum[0] / (case - 1): 1.8f} sec; \tCL: {times_sum[1] / (case - 1): 1.8f} sec')
    print(f'  - Eval 1')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[3] / (case - 1): 1.8f} sec')
    print(f'  - Eval 2')
    print(f'    Native: {times_sum[2] / (case - 1): 1.8f} sec; \tCL: {times_sum[4] / (case - 1): 1.8f} sec')
    print('--- END AvgPool')


if __name__ == '__main__':
    main()
