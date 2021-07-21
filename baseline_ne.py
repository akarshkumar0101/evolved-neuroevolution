import models_pheno
import torch
import xarray as xr
import numpy as np
import argparse

from tqdm import tqdm
from functools import partial

import util
import mnist
from ga import calc_npop_truncate

import sys

parser = argparse.ArgumentParser(description='Run basic neuroevolution.')
parser.add_argument('--n-pop', type=int, default=100)
parser.add_argument('--n-gen', type=int, default=1000)
parser.add_argument('--eps', type=float, default=1e-2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')

def run_evolution(args):
    task = mnist.MNIST()
    task.load_all_data(args.device)

    net = models_pheno.BigConvNet().to(args.device)
    triu_mask = torch.triu(torch.ones([args.n_pop]*2), diagonal=1).to(bool)
    
    pop = [torch.randn(util.count_params(net)).to(args.device)/6. for _ in range(args.n_pop)]
    fitdata = []
    loop = tqdm(range(args.n_gen), leave=False)
    for i in loop:
        fitdata.append(task.calc_pop_fitness(pop, lambda x: util.vec2model(x, net), 
                                             n_samples=1000, device=args.device))

        pop = calc_npop_truncate(pop, fitdata[-1].sel(metric='fitness').data,
                           k=30, cross_fn=util.uniform_crossover,
                           mutate_fn=partial(util.additive_noise, eps=args.eps))
        loop.set_postfix({'loss': fitdata[-1].sel(metric='loss').min().item()})
    fitdata = xr.concat(fitdata, dim='gen')
    return fitdata
    
args = parser.parse_args()
print(args)
fitdata = run_evolution(args)
torch.save(fitdata, f'data/runs/{args.seed}.ans')



