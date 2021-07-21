import models_pheno
import torch
import xarray as xr
import numpy as np
import argparse

from tqdm import tqdm

import util
import mnist
from ga import calc_npop_truncate

import sys

parser = argparse.ArgumentParser(description='Run mutation evolved neuroevolution.')
parser.add_argument('--n-pop', type=int, default=101)
parser.add_argument('--n-gen', type=int, default=150)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')


def run_evolution(args):
    task = mnist.MNIST()
    task.load_all_data(args.device)
    
    net = models_pheno.BigConvNet().to(args.device)
    triu_mask = torch.triu(torch.ones([args.n_pop]*2), diagonal=1).to(bool)
    
    pop = [torch.randn(util.count_params(net)).to(args.device)/6. for _ in range(args.n_pop)]
    mutpop = [1e-2*i for i in np.linspace(0.5, 1.5, 5)]
    
    fitdata = []
    loop = tqdm(range(args.n_gen), leave=False)
    for i in loop:
        fd = task.calc_pop_fitness(pop, lambda x: util.vec2model(x, net),
                                   n_samples=1000, device=args.device)
        fitdata.append(fd)

        idx_selected = np.argsort(fitdata[-1].sel(metric='fitness'))[::-1][:30].data
        pop_selected = [pop[i] for i in idx_selected]
        npop = [pop_selected[0]]
        
        a = np.arange(args.n_pop)//(args.n_pop-1//len(mutpop))
        mutrates = [mutpop[i] for i in a]
        
        befmut = []
        aftmut = []
        for _, mutrate in zip(range(args.n_pop-1), mutrates):
            a, b = np.random.choice(len(pop_selected), size=(2), replace=False)
            a, b = pop_selected[a], pop_selected[b]
            c = util.uniform_crossover(a, b, p=0.5)
            befmut.append(c)
            c = util.additive_noise(c, eps=1e-2)
            aftmut.append(c)
            npop.append(c.detach())
        
        fd1 = task.calc_pop_fitness(befmut, lambda x: util.vec2model(x, net),
                                   n_samples=1000, device=args.device)
        fd2 = task.calc_pop_fitness(aftmut, lambda x: util.vec2model(x, net),
                                   n_samples=1000, device=args.device)
        fd = fd2-fd1
        fd = fd.sel(metric='fitness').data.reshape(len(mutpop), -1)
        fd = fd.max(axis=-1)
        
        nmutpop = calc_npop_truncate(mutpop, fd, 
                               k=2, cross_fn=None,
                                mutate_fn=lambda x: x*np.random.uniform(.95, 1.05))
#         print((' '.join([f'{i:.04f}' for i in np.sort(mutpop)])))
#         print(fd)
#         print()
        mutpop = nmutpop
        
        pop = npop
        
        loop.set_postfix({'loss': fitdata[-1].sel(metric='loss').min().item()})
    fitdata = xr.concat(fitdata, dim='gen')
    return fitdata

args = parser.parse_args()

print(args)
fitdata = run_evolution(args)
torch.save(fitdata, f'data/runs/{args.seed}.ans')