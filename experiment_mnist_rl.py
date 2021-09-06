
import numpy as np
import scipy.stats
import xarray as xr
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import shutil
from functools import partial

from torchinfo import summary
# import torch.utils.tensorboard as tb

import models_pheno, models_decode, models_breed, models_mutate
import mnist
import ga, neuroevolution, co_neuroevolution, cmaes
import genotype
# import ordinary_ne
import util, viz
from multi_arr import MultiArr

import functions
from functions import *
import xarray as xr
import argparse
from ga import calc_npop_truncate
import optim
from viz import *
from analysis import *

def do_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
do_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from paper_meta import *
import gym


import argparse
parser = argparse.ArgumentParser('MNIST/CartPole experiments')
parser.add_argument('--env')
parser.add_argument('--algo')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_gen', default=10, type=int)
parser.add_argument('--device', default=device)
args = parser.parse_args()
print(args)
device = args.device

class MNISTNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 10)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class RLNet(nn.Module):
    def __init__(self, dim_in, dim_out, softmax=True):
        super().__init__()
        self.softmax = softmax
        
        self.lin1 = nn.Linear(dim_in, 128)
        self.lin2 = nn.Linear(128, dim_out)
        # use 2 hidden layers of 64 and tanh.
        
    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        if self.softmax:
            x = x.softmax(dim=-1)
        return x

def get_env_stats(env):
    dim_in = env.observation_space.shape[0]
    if env.action_space.dtype==int:
        dim_out = env.action_space.n
        softmax = True
    else:
        dim_out = env.action_space.shape[0]
        softmax = False
    return dim_in, dim_out, softmax


rl = True
if args.env=='mnist':
    rl = False
    task = mnist.MNIST()
    task.load_all_data(device=device)
elif args.env=='fmnist':
    rl = False
    task = mnist.MNIST(fashion=True)
    task.load_all_data(device=device)
elif args.env=='cartpole':
    env = gym.make('CartPole-v0')
elif args.env=='pendulum':
    env = gym.make('Pendulum-v0')
elif args.env=='mountain':
    env = gym.make('MountainCar-v0')
elif args.env=='acrobot':
    env = gym.make('Acrobot-v1')
    
if rl:
    stats = get_env_stats(env)
    net = RLNet(*stats)
else:
    net = MNISTNet()
    net = net.to(device)
    
    
n_dim = util.count_params(net)
def fit_mnist(pop):
    bs = pop.shape[:-1]
    pop = pop.reshape(-1, n_dim)
    pop = pop.to(device)
    fit = task.calc_pop_fitness(pop, geno2pheno=lambda x: util.vec2model(x, net), device=device)
    return torch.from_numpy(fit.sel(metric='loss').data).reshape(*bs)

def fit_rl(x, n_sample=1):
    bs = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])

    rs = []

    for xi in x:
        neti = util.vec2model(xi, net)
        for seed in range(n_sample):
            r = 0
            obs = env.reset()
            for _ in range(600):
                inp = torch.from_numpy(obs).reshape(1, -1).float().to(x.device)
                out = neti(inp).detach().cpu().numpy()[0]
                if stats[2]:
                    action = np.random.choice(len(out), size=1, p=out).item()
                else:
                    action = out
                obs, reward, done, info = env.step(action)
                r += reward
                if(done):
                    break
            rs.append(r)
    fits = -torch.tensor(rs).reshape(*bs, n_sample).mean(dim=-1).to(x)
    return fits


if rl:
    optim_fn = partial(fit_rl, n_sample=5)
else:
    optim_fn = fit_mnist

algo2algo_fn['gsmr'] = partial(algo2algo_fn['gsmr'], n_mutpop=10)

algo2algo_fn['fmr'] = re_fmr

def run_experiment(seed, algo, n_gen):
    config = {'seed': seed, 'algo': algo, 'n_gen': n_gen}
    
    do_seed(seed)
    pop = torch.randn(101, n_dim)/10.
    a = algo2algo_fn[algo](pop, optim_fn, n_gen, tqdm=tqdm)
    pops, fits, mrs = a[:3]
    fits = fits.min(dim=-1).values
    while mrs.ndim>1:
        mrs = mrs.log().mean(dim=-1).exp()
    
    data = {'bestpop': pops[-1, 0], 'fits': fits, 'mrs': mrs}
    
    folder = f'/work/08258/akumar01/maverick2/evolved-neuroevolution/data/{args.env}'
    torch.save(config, f'{folder}/config_{algo}_{seed}.pt')
    torch.save(data, f'{folder}/data_{algo}_{seed}.pt')

run_experiment(args.seed, args.algo, args.n_gen)
