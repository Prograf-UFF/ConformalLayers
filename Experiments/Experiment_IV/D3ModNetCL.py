import torch
import os
import sys
import numpy as np
import argparse
import time
from resource import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Experiments.networks.dknet import D3ModNetCL


@torch.no_grad()
def test(net, iteration, data, depth, batch_size, device):
    net.eval()
    pid = os.getpid()
    f = open('D3ModNetCL_results.csv', 'a')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        os.system("echo 5 >> /proc/{}/clear_refs".format(pid))
        start = time.time()
    net(data)
    if device.type == 'cuda':
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        c = 0
        m = torch.cuda.max_memory_allocated(device=device)
    else:
        end = time.time()
        t = end-start
        c = 0
        m = getrusage(RUSAGE_SELF).ru_maxrss / 1024
    line = '{},{},{},{},{},{}'.format(iteration, depth, batch_size, t, c, m)
    f.write(line + '\n')
    print(line)
    f.close()
    

def main():
    # Device parameters
    device = torch.device('cuda:{}'.format(os.environ['GPU'])) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        print('Warning: The device was set to CPU.')

    # Set the seeds for reproducibility
    torch.manual_seed(1992)
    np.random.seed(1992)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inferences', help="num of inferences", default=30, type=int)
    parser.add_argument('--batch_size', help="batch_size", default=10, type=int)
    parser.add_argument('--depth', help="depth", default=3, type=int)
    args = parser.parse_args()

    # Fake data generation
    data = torch.rand((args.batch_size, 3, 32, 32), device=device)

    # Network instance
    net = D3ModNetCL().to(device)

    # Inference main loop
    for inference_idx in range(args.num_inferences):
        test(net, inference_idx, data, args.depth, args.batch_size, device)


if __name__ == "__main__":
    main()
