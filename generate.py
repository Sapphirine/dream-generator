import subprocess
import sys


model = sys.argv[1]
gpu = sys.argv[2]

result = subprocess.run(['python', 'extern/pytorch-CycleGAN-and-pix2pix/test.py', '--dataroot', 'samples',
                         '--name', model, '--model', 'test', '--no_dropout', '--gpu_ids',
                         gpu, '--aspect_ratio', '0.75'])


