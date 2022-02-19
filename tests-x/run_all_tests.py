import subprocess, sys

subprocess.run([sys.executable, 'test-AvgPool.py'])
subprocess.run([sys.executable, 'test-CL.py'])
subprocess.run([sys.executable, 'test-Conv.py'])
subprocess.run([sys.executable, 'test-Flatten.py'])
