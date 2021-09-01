import torch
import optim

from functools import partial

algos = ['ofmr', 'lamr_100', '1cmr', '15mr', 'ucb_5', 'ucb_10', 'nsmr', 'gsmr']
def subtract_list(a, b):
    return [i for i in a if i not in b]
algo_ours = 'gsmr'
algos_oracle = ['ofmr', 'lamr_100']
algos_normal = [a for a in algos if a not in algos_oracle]
labels = ['*OFMR', '*LAMR-100', '1CMR', '15MR', 'UCB/5', 'UCB/10', 'NSMR', 'GSMR']
# colors = ['cornflowerblue', 'yellow', 'cyan', 'darkblue', 'purple', 'magenta', 'darkgreen', 'darkred']
colors = ['orangered', 'red', 'cyan', 'darkblue', 'purple', 'magenta', 'darkgreen', 'black']
algo2color = {algo: color for algo, color in zip(algos, colors)}
algo2label = {algo: label for algo, label in zip(algos, labels)}

log_mr_low, log_mr_high = -3, 0
mrs_grid_5 = torch.logspace(log_mr_low, log_mr_high, 5)
mrs_grid_10 = torch.logspace(log_mr_low, log_mr_high, 10)
mrs_grid_20 = torch.logspace(log_mr_low, log_mr_high, 20)

re_ofmr = partial(optim.run_evolution_ofmr, mrs=mrs_grid_10, n_sample=1)

re_la_10 = partial(optim.run_evolution_look_ahead, 
                    mrs=mrs_grid_10, look_ahead=10, every_k_gen=10, n_sims=1)
re_la_100 = partial(optim.run_evolution_look_ahead, 
                    mrs=mrs_grid_10, look_ahead=100, every_k_gen=100, n_sims=1)

def re_1c(pop, optim_fn, n_gen, k=.5, k_elite=None):
    mr = torch.tensor(1./pop.shape[-1]).to(pop)
    a = optim.run_evolution_base(pop, optim_fn, n_gen, mr, k=k, k_elite=k_elite)
    return list(a)+[mr.repeat(n_gen+1)]

re_15 = partial(optim.run_evolution_one_fifth, 
                mr=1e-2, mr_mut=1.01, thresh=0.2)

re_ucb_5 = partial(optim.run_evolution_ucb, mrs=mrs_grid_5)
re_ucb_10 = partial(optim.run_evolution_ucb, mrs=mrs_grid_10)

re_nsmr = partial(optim.run_evolution_ns,
                  mr=None, mr_mut=2.0)

re_gsmr = partial(optim.run_evolution_ours, 
                  n_mutpop=10, mr=None, mr_mut=2.0)

algo_fns = [re_ofmr, re_la_100, re_1c, re_15, re_ucb_5, re_ucb_10, re_nsmr, re_gsmr]
algo2algo_fn = {algo: algo_fn for algo, algo_fn in zip(algos, algo_fns)}