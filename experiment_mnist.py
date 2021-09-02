
import numpy as np
import scipy.stats
import xarray as xr
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

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


task = mnist.MNIST()
task.load_all_data()


class Net(nn.Module):
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
n_dim = util.count_params(Net())

device = 'cuda:0'
net = models_pheno.BigConvNet()
net = net.to(device)
def fit_mnist(pop):
    bs = pop.shape[:-1]
    pop = pop.reshape(-1, n_dim)
    pop = pop.to(device)
    fit = task.calc_pop_fitness(pop, geno2pheno=lambda x: util.vec2model(x, net), device=device)
    return torch.from_numpy(fit.sel(metric='loss').data).reshape(*bs)
optim_fn = fit_mnist

def run_experiment_mnist(seed, algo):
    config = {'seed': seed, 'algo': algo}
    
    do_seed(seed)
    pop = torch.randn(101, n_dim)/10.
    a = algo2algo_fn[algo](pop, fit_mnist, 1000)
    pops, fits, mrs = a[:3]
    
    data = {'pops': pops, 'fits': fits, 'mrs': mrs}
    torch.save(config, f'/work/08258/akumar01/maverick2/evolved-neuroevolution/data/mnist/config_{algo}_{seed}.pt')
    torch.save(data, f'/work/08258/akumar01/maverick2/evolved-neuroevolution/data/mnist/data_{algo}_{seed}.pt')


for seed in tqdm(range(5), leave=True):
    for algo in tqdm(algos_normal, leave=False):
        run_experiment_mnist(seed, algo)

