try:
    import cl
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import cl

import time, torch, warnings

# Device to run the workload
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.set_device(DEVICE) if DEVICE.type == 'cuda' else warnings.warn(f'The device was set to {DEVICE}.', RuntimeWarning)

def main():
    tol = 1e-6
    print('--- START CL')
    # Input data
    input = torch.as_tensor([[[76, 29, 21, -61, 15, -6, -16, -26, 54, 53, -38, -14, -19, -85, 99, 38, -48, 25, -89, 8]]], dtype=torch.float32, device=DEVICE)
    # Layers
    layers = [
        # Layer 1
        [cl.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=3, dilation=1),
         cl.AvgPool1d(kernel_size=2, stride=2, padding=0),
         cl.SRePro(alpha=10)],
        # Layer 2
        [cl.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1),
         cl.AvgPool1d(kernel_size=2, stride=2, padding=0),
         cl.SRePro(alpha=20)],
        # Layer 3
        [cl.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=1, dilation=1),
         cl.AvgPool1d(kernel_size=3, stride=2, padding=0),
         cl.SRePro(alpha=30)]]
    layers[0][0].weight.data.copy_(torch.as_tensor([7, -8, -8, 10, 9], dtype=torch.float32).view(1, 1, 5))
    layers[1][0].weight.data.copy_(torch.as_tensor([2, 8, 3], dtype=torch.float32).view(1, 1, 3))
    layers[2][0].weight.data.copy_(torch.as_tensor([8, -5, 0, -7], dtype=torch.float32).view(1, 1, 4))
    # Expected output
    expected = [
        torch.as_tensor([[[0.0025009, -0.00426723, 0.00260744, -0.00108223, 0.0032607, -0.00447751, -0.000361677, 0.00495975, -0.00483358, -0.000314014, 0.000381303]]], dtype=torch.float32, device=DEVICE),
        torch.as_tensor([[[-0.000517973, 0.000566045, -0.000730433, 0.00100908, -0.00149525]]], dtype=torch.float32, device=DEVICE),
        torch.as_tensor([[[0.000261716]]], dtype=torch.float32, device=DEVICE)]
    # Build newtworks using layers from 1 to k, where k = 1, 2, 3
    for k in range(1, len(layers) + 1):
        modules = []
        for l in range(k):
            modules += layers[l]
        net = cl.ConformalLayers(*modules).to(DEVICE)
        #
        net.train()
        start_time = time.time()
        output_train = net(input)
        train_time = time.time() - start_time
        if torch.max(torch.abs(expected[k-1] - output_train)) > tol:
            raise RuntimeError(f'expected = {expected[k-1]}\n  output_train = {output_train}')
        #
        net.eval()
        start_time = time.time()
        output_eval1 = net(input)
        eval1_time = time.time() - start_time
        if torch.max(torch.abs(expected[k-1] - output_eval1)) > tol:
            raise RuntimeError(f'expected = {expected[k-1]}\n  output_eval1 = {output_eval1}')
        #
        net.eval()
        start_time = time.time()
        output_eval2 = net(input)
        eval2_time = time.time() - start_time
        if torch.max(torch.abs(expected[k-1] - output_eval2)) > tol:
            raise RuntimeError(f'expected = {expected[k-1]}\n  output_eval2 = {output_eval2}')
        print(f'  --- Case {k}')
        print(f'    Train : {train_time: 1.8f} sec')
        print(f'    Eval 1: {eval1_time: 1.8f} sec')
        print(f'    Eval 2: {eval2_time: 1.8f} sec')
    print('--- END CL')


if __name__ == '__main__':
    main()
