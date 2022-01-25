
from functions import *
from paper_meta import *

import copy
import xarray as xr
import numpy as np
import torch
from tqdm import tqdm


def do_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)



algos = copy.deepcopy(algos_all)
optim_fns = [ackley, griewank, rastrigin, rosenbrock, sphere, lin_fn]
# n_dims = [2, 30, 100, 1000]
# n_dims = [1000, 100, 30, 2]
# n_dims = [10]
n_dims = [100]

# init_pop_vars = [.1, 1, 5, 10]
# init_pop_vars = [1, 10]
init_pop_vars = [10]

n_seed = 30
n_seed = 2
# n_gen = 1000
n_gen = 300
n_pop = 101

data = np.empty((len(algos), len(optim_fns), len(n_dims), 
                 len(init_pop_vars), n_seed, 4), dtype=object)
data = xr.DataArray(data, dims=('algo', 'optim_fn', 'n_dim', 
                                'init_pop_var', 'seed', 'fits_mrs'),
                    coords={'algo': algos, 
                            'optim_fn': optim_fns,
                            'n_dim': n_dims,
                            'init_pop_var': init_pop_vars,
                            'seed': list(range(n_seed)),
                            'fits_mrs': ['fits', 'mrs', 'best_fit', 'avg_fit'],
                           })

pbar = tqdm(total=len(optim_fns)*len(n_dims)*len(init_pop_vars)*n_seed*len(algos))
for optim_fn in optim_fns:
    for n_dim in n_dims:
        for init_pop_var in init_pop_vars:
            for seed in range(n_seed):
                for algo in algos:
                    do_seed(seed)
                    pop = torch.randn(n_pop, n_dim)*init_pop_var
                    real_n_gen = 100 if optim_fn==lin_fn else n_gen
                    # elif n_dim==2:
                    #     n_gen = 10
                    # elif n_dim==30:
                    #     n_gen = 30
                    # elif n_dim==100:
                    #     n_gen = 10
                    # elif n_dim==1000:
                    #     n_gen = 25
                    res = algo2algo_fn[algo](pop, optim_fn, real_n_gen)
                    pops, fits, mrs = res[:3]
                    fits = fits.min(dim=-1).values
                    while mrs.ndim>1:
                        mrs = mrs.log().mean(dim=-1).exp()
                    
                    a = data.sel(algo=algo, optim_fn=optim_fn, n_dim=n_dim, 
                                 init_pop_var=init_pop_var, seed=seed)
#                     print(a[0].data)
                    # a.data[0] = fits.detach().cpu().numpy()
#                     print(a[0].data)
                    # a.data[1] = mrs.detach().cpu().numpy()
                    a.data[2] = fits.min().item()
                    a.data[3] = fits.mean().item()
                    
                    pbar.update(n=1); pbar.refresh()
torch.save(data, 'results/main_data.th')