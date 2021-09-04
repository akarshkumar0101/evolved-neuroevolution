
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
mnistnet = MNISTNet()
CartPoleNet = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), 
                            nn.Linear(128, 2), nn.Softmax(dim=-1))

env2net = {'mnist': mnistnet, 'cartpole': CartPoleNet}
net = env2net[args.env]
n_dim = util.count_params(net)
net = net.to(device)


def fit_mnist(pop):
    bs = pop.shape[:-1]
    pop = pop.reshape(-1, n_dim)
    pop = pop.to(device)
    fit = task.calc_pop_fitness(pop, geno2pheno=lambda x: util.vec2model(x, net), device=device)
    return torch.from_numpy(fit.sel(metric='loss').data).reshape(*bs)

def fit_cartpole(x, n_sample=1):
    x = x.to(args.device)
    bs = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    
    rs = []
    
    for xi in x:
        neti = util.vec2model(xi, net)
        for seed in range(n_sample):
            r=0
            env = gym.make("CartPole-v0")
            obs = env.reset()
            for _ in range(250):
                inp = torch.from_numpy(obs).reshape(1, -1).float().to(args.device)
                out = neti(inp).detach().cpu().numpy()[0]
                action = np.random.choice(range(2), size=1, p=out).item()
                obs, reward, done, info = env.step(action)
                r += reward
                if(done):
                    break
            rs.append(r)
    rs = torch.tensor(rs).reshape(*bs, n_sample).mean(dim=-1).to(x)
    return -rs

if args.env=='mnist':
    optim_fn = fit_mnist
elif args.env=='cartpole':
    optim_fn = partial(fit_cartpole, n_sample=5)

print(algo2algo_fn['gsmr'])
algo2algo_fn['gsmr'] = partial(algo2algo_fn['gsmr'], n_mutpop=10)

algo2algo_fn['fmr'] = re_fmr
print(algo2algo_fn['gsmr'])

def run_experiment_mnist(seed, algo, n_gen):
    config = {'seed': seed, 'algo': algo, 'n_gen': n_gen}
    
    do_seed(seed)
    pop = torch.randn(101, n_dim)/10.
    a = algo2algo_fn[algo](pop, optim_fn, n_gen, tqdm=tqdm)
    pops, fits, mrs = a[:3]
    
    data = {'pops': pops, 'fits': fits, 'mrs': mrs}
    torch.save(config, f'/work/08258/akumar01/maverick2/evolved-neuroevolution/data/{args.env}/config_{algo}_{seed}.pt')
    torch.save(data, f'/work/08258/akumar01/maverick2/evolved-neuroevolution/data/{args.env}/data_{algo}_{seed}.pt')

run_experiment_mnist(args.seed, args.algo, args.n_gen)
