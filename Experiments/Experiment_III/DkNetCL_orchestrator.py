import tqdm
import subprocess

fname = 'DkNetCL_results.csv'
f = open(fname, 'w')
f.write('iteration,depth,batch_size,time,cpu_usage,memory_usage\n')
f.close()

depth = list(range(1, 13))
for i in tqdm.tqdm(depth, total=len(depth)):
    subprocess.run(['python', 'DkNetCL.py', '--num_inferences=100', '--batch_size=64', '--depth={}'.format(i)])
