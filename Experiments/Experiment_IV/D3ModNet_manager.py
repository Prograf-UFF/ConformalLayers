import tqdm
import subprocess

fname = 'D3ModNet_results.csv'
f = open(fname, 'w')
f.write('iteration,batch_size,time,cpu_usage,memory_usage\n')
f.close()

batches = list(range(1000, 100000, 1000))
for i in tqdm.tqdm(batches, total=len(batches)):
    subprocess.run(['python', 'D3ModNet.py', '--num_inferences=100', '--batch_size={}'.format(i)])
