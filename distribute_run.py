import subprocess, os, sys

import argparse

# import torch
import numpy as np
# import xarray as xr

parser = argparse.ArgumentParser(description='Run distributed session')
parser.add_argument('--seeds', nargs='+', default=[0])
parser.add_argument('--devices', nargs='+', default=['cpu'])

# args = parser.parse_args()
args, ext = parser.parse_known_args()
# print(sys.argv)
# print(args)
# print(ext)

if len(args.seeds)==2:
    args.seeds = np.arange(int(args.seeds[0]), int(args.seeds[1]))
elif len(args.seeds)==1:
    args.seeds = np.arange(int(args.seeds[0]))
    
args.devices = args.devices*(len(args.seeds)//len(args.devices)+1)

procs = []
for seed, device in zip(args.seeds, args.devices):
    print(seed, device)
    command = ' '.join(ext)
    command = f'python3 ' + command + f' --seed {seed} --device {device}'
    print(command)
#     subprocess.call(command.split(' '))
    procs.append(subprocess.Popen(command.split(' ')))
    
for proc in procs:
    proc.communicate()
    

