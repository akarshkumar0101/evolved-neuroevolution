import numpy as np
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

# algos = ['ofmr', 'lamr_100', '1cmr', '15mr', 'ucb_5', 'ucb_10', 'nsmr', 'gsmr']
# labels = ['OFMR', 'LAMR-100', '1CMR', '15MR', 'UCB/5', 'UCB/10', 'NSMR', 'GSMR']
# colors = ['cornflowerblue', 'yellow', 'cyan', 'pink', 'purple', 'magenta', 'green', 'red']
# algo2color = {algo: color for algo, color in zip(algos, colors)}
# algo2label = {algo: label for algo, label in zip(algos, labels)}

# log_mr_low, log_mr_high = -3, 0
# mrs_grid_5 = torch.logspace(log_mr_low, log_mr_high, 5)
# mrs_grid_10 = torch.logspace(log_mr_low, log_mr_high, 10)
# mrs_grid_20 = torch.logspace(log_mr_low, log_mr_high, 20)

# re_ofmr = partial(optim.run_evolution_ofmr, mrs=mrs_grid_10, n_sample=1)

# re_la_10 = partial(optim.run_evolution_look_ahead, 
#                     mrs=mrs_grid_10, look_ahead=10, every_k_gen=10, n_sims=1)
# re_la_100 = partial(optim.run_evolution_look_ahead, 
#                     mrs=mrs_grid_10, look_ahead=100, every_k_gen=100, n_sims=1)

# def re_1c(pop, optim_fn, n_gen, k=.5, k_elite=None):
#     mr = torch.tensor(1./pop.shape[-1]).to(pop)
#     a = optim.run_evolution_base(pop, optim_fn, n_gen, mr, k=k, k_elite=k_elite)
#     return list(a)+[mr.repeat(n_gen+1)]

# re_15 = partial(optim.run_evolution_one_fifth, 
#                 mr=1e-2, mr_mut=1.01, thresh=0.2)

# re_ucb_5 = partial(optim.run_evolution_ucb, mrs=mrs_grid_5)
# re_ucb_10 = partial(optim.run_evolution_ucb, mrs=mrs_grid_10)

# re_nsmr = partial(optim.run_evolution_ns,
#                   mr=None, mr_mut=2.0)

# re_gsmr = partial(optim.run_evolution_ours, 
#                   n_mutpop=10, mr=None, mr_mut=2.0)


# re_cmaes = partial(optim.run_evolution_cmaes, mr=1e-2)
# # def run_evolution_cmaes(pop, optim_fn, n_gen, mr, tqdm=None):
# # def run_evolution_ours(pop, optim_fn, n_gen

# algo_fns = [re_ofmr, re_la_100, re_1c, re_15, re_ucb_5, re_ucb_10, re_nsmr, re_gsmr]
# algo2algo_fn = {algo: algo_fn for algo, algo_fn in zip(algos, algo_fns)}


from paper_meta import *

if __name__ == '__main__':
    # optim_fns = [ackley, rastrigin, rosenbrock, sphere, lin_fn, griewank]
    # n_dims = [2, 30, 100, 1000]
    n_dims = [1000, 100, 30, 2]
    # n_dims = [100]

    # init_pop_vars = [.1, 1, 5, 10]
    init_pop_vars = [1, 10]
    # init_pop_vars = [10]

    n_seed = 40
    # algos = ['1cmr', '15mr', 'ucb_5', 'ucb_10', 'nsmr', 'gsmr']
    algos = algos_all # ['ofmr', 'lamr_100', '1cmr', '15mr', 'ucb_5', 'ucb_10', 'nsmr', 'gsmr', 'gsmr_avg', 'fmr', 'gsmr_fix']

    n_gen = 1000

    n_pop = 101
    data = np.zeros((len(algos), len(optim_fns), len(n_dims), 
                    len(init_pop_vars), n_seed, n_gen+1, 2))
    data = xr.DataArray(data, dims=('algo', 'optim_fn', 'n_dim', 
                                    'init_pop_var', 'seed', 'gen', 'fits_mrs'),
                        coords={'algo': algos, 
                                'optim_fn': optim_fns,
                                'n_dim': n_dims,
                                'init_pop_var': init_pop_vars,
                                'seed': list(range(n_seed)),
                                'gen': list(range(n_gen+1)),
                                'fits_mrs': ['fits', 'mrs'],
                            })

    pbar = tqdm(total=len(optim_fns)*len(n_dims)*len(init_pop_vars)*n_seed*len(algos))
    for optim_fn in optim_fns:
        for n_dim in n_dims:
            for init_pop_var in init_pop_vars:
                for seed in range(n_seed):
                    for algo in algos:
                        do_seed(seed)
                        pop = torch.randn(n_pop, n_dim)*init_pop_var
                        res = algo2algo_fn[algo](pop, optim_fn, n_gen)
                        pops, fits, mrs = res[:3]
                        fits = fits.min(dim=-1).values
                        while mrs.ndim>1:
                            mrs = mrs.log().mean(dim=-1).exp()
                        
                        a = data.sel(algo=algo, optim_fn=optim_fn, n_dim=n_dim, 
                                    init_pop_var=init_pop_var, seed=seed)
                        
                        a.sel(fits_mrs='fits')[:] = fits.detach().cpu().numpy()
                        a.sel(fits_mrs='mrs')[:] = mrs.detach().cpu().numpy()
                
                        pbar.update(n=1); pbar.refresh()
    torch.save(data, 'results/main_data_griewank.th')
