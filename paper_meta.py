import torch
import optim

from functools import partial

from functions import *

optim_fn2label = {
    ackley: 'Ackley',
    griewank: 'Griewank',
    rastrigin: 'Rastrigin',
    rosenbrock: 'Rosenbrock',
    sphere: 'Sphere',
    lin_fn: 'Linear',
}

def subtract_list(a, b):
    return [i for i in a if i not in b]

algos_all = ['ofmr', 'lamr_100', '1cmr', '15mr', 'ucb_5', 'ucb_10', 'nsmr', 'gsmr', 'gsmr_avg', 'fmr', 'gsmr_fix']
algo_ours = 'gsmr'
algos_oracle = ['ofmr', 'lamr_100']
algos_normal = [a for a in algos_all if a not in algos_oracle]
labels = ['$^\dagger$OFMR', '$^\dagger$LAMR-100', '1CMR', '15MR', 'UCB/5', 'UCB/10', 'SAMR', 'GESMR', 'GESMR-AVG', 'FMR', 'GESMR-FIX']
# colors = ['cornflowerblue', 'yellow', 'cyan', 'darkblue', 'purple', 'magenta', 'darkgreen', 'darkred']
colors = ['maroon', 'red', 'cyan', 'darkblue', 'purple', 'magenta', 'darkgreen', 'black', 'yellow', 'slategrey', 'olive']
algo2color = {algo: color for algo, color in zip(algos_all, colors)}
algo2label = {algo: label for algo, label in zip(algos_all, labels)}

log_mr_low, log_mr_high = -3, 0
mrs_grid_5 = torch.logspace(log_mr_low, log_mr_high, 5)
mrs_grid_10 = torch.logspace(log_mr_low, log_mr_high, 10)
mrs_grid_20 = torch.logspace(log_mr_low, log_mr_high, 20)

re_ofmr = partial(optim.run_evolution_ofmr, mrs=mrs_grid_10, n_sample=1)

re_la_10 = partial(optim.run_evolution_look_ahead,
                    mrs=mrs_grid_10, look_ahead=10, every_k_gen=10, n_sims=1)
re_la_100 = partial(optim.run_evolution_look_ahead, 
                    mrs=mrs_grid_10, look_ahead=100, every_k_gen=100, n_sims=1)

def re_fmr(pop, optim_fn, n_gen, mr=1e-2, k=.5, k_elite=None, tqdm=lambda x: x):
    mr = torch.tensor(mr).to(pop)
    a = optim.run_evolution_base(pop, optim_fn, n_gen, mr, k=k, k_elite=k_elite, tqdm=tqdm)
    return list(a)+[mr.repeat(n_gen+1)]

def re_1c(pop, optim_fn, n_gen, k=.5, k_elite=None, tqdm=lambda x: x):
    mr = torch.tensor(1./pop.shape[-1]).to(pop)
    a = optim.run_evolution_base(pop, optim_fn, n_gen, mr, k=k, k_elite=k_elite, tqdm=tqdm)
    return list(a)+[mr.repeat(n_gen+1)]

re_15 = partial(optim.run_evolution_one_fifth, 
                mr=1e-2, mr_mut=2.0, thresh=0.2)

re_ucb_5 = partial(optim.run_evolution_ucb, mrs=mrs_grid_5)
re_ucb_10 = partial(optim.run_evolution_ucb, mrs=mrs_grid_10)

re_nsmr = partial(optim.run_evolution_ns,
                  mr=None, mr_mut=2.0)

re_gsmr = partial(optim.run_evolution_ours, 
                  n_mutpop=10, mr=None, mr_mut=2.0)
re_gsmr_avg = partial(optim.run_evolution_ours, 
                  n_mutpop=10, mr=None, mr_mut=2.0, useavg=True)
re_gsmr_fix = partial(optim.run_evolution_ours, 
                  n_mutpop=10, mr=None, mr_mut=2.0, fixedmrpop=True)

algo_fns = [re_ofmr, re_la_100, re_1c, re_15, re_ucb_5, re_ucb_10, re_nsmr, re_gsmr, re_gsmr_avg, re_fmr, re_gsmr_fix]
algo2algo_fn = {algo: algo_fn for algo, algo_fn in zip(algos_all, algo_fns)}
