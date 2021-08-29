import tqdm
import subprocess

fname = 'DkNet_results.csv'
f = open(fname, 'w')
f.write('iteration,batch_size,time,cpu_usage,memory_usage\n')
f.close()

depth = list(range(1, 13))
for i in tqdm.tqdm(depth, total=len(depth)):
    subprocess.run(['python', 'DkNet.py', '--num_inferences=100', '--batch_size=64', '--depth={}'.format(i)])
