import torch
import numpy as np
import cma

def calc_npop_idxs(fit, k=.5, k_elite=None):
    k = int(len(fit)*k)
    if k_elite is None:
        k_elite = 1
    else:
        k_elite = int(len(fit)*k_elite)
    idxs_sort = torch.argsort(fit)
    idxs_elite = idxs_sort[:k_elite]
    idxs_rest = idxs_sort[:k]
    n_children = len(fit)-k_elite
    idxs_rest = idxs_rest[np.random.choice(len(idxs_rest), size=(n_children, ), replace=True)]
    idxs_cat = torch.cat([idxs_elite, idxs_rest], dim=-1)
    return idxs_elite, idxs_rest, idxs_cat

def calc_npop_gaussian(pop, mr, 
                       idxs_elite, idxs_rest, idxs_cat=None):
    npop = torch.zeros_like(pop)
    npop[:len(idxs_elite)] = pop[idxs_elite]
    p = pop[idxs_rest]
    eps = torch.randn_like(p)
    npop[len(idxs_elite):] = p+eps*mr
    return npop

def calc_npop_gaussian_mean_std(pop, mr_mean, mr_std, 
                                idxs_elite, idxs_rest, idxs_cat=None):
    npop = torch.zeros_like(pop)
    npop[:len(idxs_elite)] = pop[idxs_elite]
    p = pop[idxs_rest]
    eps = torch.randn_like(p)
    npop[len(idxs_elite):] = p+(eps*mr_std+mr_mean)
    return npop


def calc_npop_log_uniform(pop, mr, idxs_elite, idxs_rest, idxs_cat=None):
    npop = torch.zeros_like(pop)
    npop[:len(idxs_elite)] = pop[idxs_elite]
    p = pop[idxs_rest]
#     print(idxs_elite, idxs_rest)
    eps = -1+2*torch.rand_like(p)
#     print(eps.shape, p.mean(), mr, eps.mean())
    npop[len(idxs_elite):] = p*(mr**eps)
    return npop

def run_evolution_base(pop, optim_fn, n_gen, mr, 
                       k=.5, k_elite=None,
                       tqdm=lambda x: x):
    data = []
    fit = optim_fn(pop)
    data.append((pop, fit))
    for i in tqdm(range(n_gen)):
        pop = calc_npop_gaussian(pop, mr, *calc_npop_idxs(fit, k=k, k_elite=k_elite))
        fit = optim_fn(pop)
        data.append((pop, fit))
    # pops, fits
    return [torch.stack([d[i] for d in data]) for i in range(len(data[0]))]

def run_evolution_ns(pop, optim_fn, n_gen, mr=None, mr_mut=2., 
                     k=.5, k_elite=None,
                     tqdm=lambda x: x):
    if mr is None:
        mutpop = torch.logspace(-3, 3, len(pop), device=pop.device)[:, None]
    else:
        mutpop = torch.linspace(mr, mr, len(pop), device=pop.device)[:, None]
        
    data = []
    fit = optim_fn(pop)
    data.append((pop, fit, mutpop))
    for i in tqdm(range(n_gen)):
        idxs_elite, idxs_rest, idxs_cat = calc_npop_idxs(fit, k=k, k_elite=k_elite)
        pop = calc_npop_gaussian(pop, mutpop[idxs_rest], idxs_elite, idxs_rest, idxs_cat)
        mutpop = calc_npop_log_uniform(mutpop, mr_mut, idxs_elite, idxs_rest, idxs_cat)
        fit = optim_fn(pop)
        data.append((pop, fit, mutpop))
        
    # pops, fits, mrs, fitmrs
    return [torch.stack([d[i] for d in data]) for i in range(len(data[0]))]


def run_evolution_ours(pop, optim_fn, n_gen, n_mutpop=10, mr=None, mr_mut=2., 
                       k=.5, k_elite=None,
                       tqdm=lambda x: x, k_mr=.5):
    if mr is None:
        mutpop = torch.logspace(-3, 3, n_mutpop, device=pop.device)[:, None]
    else:
        mutpop = torch.linspace(mr, mr, n_mutpop, device=pop.device)[:, None]
        
    data = []
    mrpop2mr = np.arange(len(pop)-1)//int(len(pop)/len(mutpop))
    fit = optim_fn(pop)
    for i in tqdm(range(n_gen)):
        bpop = pop
        bfit = fit
        
        idxs_elite, idxs_rest, idxs_cat = calc_npop_idxs(fit, k=k, k_elite=k_elite)
        pop = calc_npop_gaussian(pop, mutpop[mrpop2mr], idxs_elite, idxs_rest, idxs_cat)
        fit = optim_fn(pop)
        fit_mrs = (fit[1:]-bfit[idxs_rest]).reshape(len(mutpop), -1)
        data.append((bpop, bfit, mutpop, fit_mrs))
        fit_mrs_min = fit_mrs.min(dim=-1).values
        mutpop = calc_npop_log_uniform(mutpop, mr_mut, *calc_npop_idxs(fit_mrs_min, k=k_mr))
    data.append((pop, fit, mutpop, fit_mrs))
        
    # pops, fits, mrs, fitmrs
    return [torch.stack([d[i] for d in data]) for i in range(len(data[0]))]

def calc_ofmr(pop, optim_fn, n_gen, mrs, n_sample=5,
              k=.5, k_elite=None, tqdm=lambda x: x):
    f = []
    for mr in tqdm(mrs):
        for i in range(n_sample):
            pops, fits = run_evolution_base(pop, optim_fn, n_gen, mr=mr)
            f.append(fits)
    f = torch.stack(f).reshape(len(mrs), n_sample, n_gen+1, pop.shape[-2])
    return mrs[f.min(dim=-1).values[..., -1].mean(dim=-1).argmin()], f

def run_evolution_look_ahead(pop, optim_fn, n_gen, 
                             mrs, look_ahead, every_k_gen, n_sims,
                             k=.5, k_elite=None,
                             tqdm=lambda x: x):
    data = []
    fit = optim_fn(pop)
    
    mr_best, _ = calc_ofmr(pop, optim_fn, look_ahead, mrs, n_sample=n_sims,
                        k=k, k_elite=k_elite)
    data.append((pop, fit, mr_best))
    for i in tqdm(range(n_gen)):
        if i%every_k_gen==0:
            mr_best, _ = calc_ofmr(pop, optim_fn, look_ahead, mrs, n_sample=n_sims,
                                k=k, k_elite=k_elite)
            
        pop = calc_npop_gaussian(pop, mr_best, *calc_npop_idxs(fit, k=k, k_elite=k_elite))
        fit = optim_fn(pop)
        data.append((pop, fit, mr_best))
    # pops, fits
    return [torch.stack([d[i] for d in data]) for i in range(len(data[0]))]
def run_evolution_one_fifth(pop, optim_fn, n_gen, mr, mr_mut=1.01, thresh=1/5.,
                            k=.5, k_elite=None,
                            tqdm=lambda x: x):
    data = []
    fit = optim_fn(pop)
    mr = torch.tensor(mr) if type(mr) is float else mr
    data.append((pop, fit, mr))
    for i in tqdm(range(n_gen)):
        idxs_elite, idxs_rest, idxs_cat = calc_npop_idxs(fit, k=k, k_elite=k_elite)
        npop = calc_npop_gaussian(pop, mr, idxs_elite, idxs_rest, idxs_cat)
        nfit = optim_fn(npop)
        
        dfit = nfit[1:]-fit[idxs_rest]
        if (dfit<0).sum()/len(dfit) > thresh:
            mr = mr*mr_mut
        else:
            mr = mr/mr_mut
        
        pop = npop
        fit = nfit
        data.append((pop, fit, mr))
    # pops, fits, mrs
    return [torch.stack([d[i] for d in data]) for i in range(len(data[0]))]

def run_evolution_cmaes(pop, optim_fn, n_gen, mr, tqdm=None):
    es = cma.CMAEvolutionStrategy(pop.mean(dim=-1).tolist(), mr)
    data = []
    loop = range(n_gen)
    if tqdm is not None: loop = tqdm(loop)
    for i in loop:
        solutions = es.ask()
        pop = torch.from_numpy(np.stack(solutions))
        fit = optim_fn(pop)
        es.tell(solutions, fit.tolist())
        data.append((pop, fit))
        
    pops = torch.stack([d[0] for d in data])
    fits = torch.stack([d[1] for d in data])
    return pops, fits
    
    
class UCBController():
    def __init__(self, arms, c=0.5):
        self.arms = arms
        self.rewards = [[] for _ in self.arms]
        self.Ns = np.zeros(len(self.arms))
        self.c = c
        
    def sample(self):
        if (self.Ns==0).any():
            i = (self.Ns==0).argmax()
        else:
            Qs = np.array([np.array(r)[-10:].mean() for r in self.rewards])

            t = self.Ns.sum()

            explore = np.sqrt(np.log(t)/self.Ns)
            exploit = Qs

            i = np.argmax(exploit+self.c*explore)
        return i, self.arms[i]
    
    def report_reward(self, i, reward):
        self.Ns[i] += 1
        self.rewards[i].append(reward)
    
    
def run_evolution_ucb(pop, optim_fn, n_gen, mrs, 
                      k=.5, k_elite=None, 
                      return_ucb=False, 
                      tqdm=lambda x: x):
    data = []
    fit = optim_fn(pop)
    mr = mrs[0]

    ucb = UCBController(mrs)

    data.append((pop, fit, mr))
    for i in tqdm(range(n_gen)):
        idxs_elite, idxs_rest, idxs_cat = calc_npop_idxs(fit, k=k, k_elite=k_elite)
        npop = calc_npop_gaussian(pop, mr, idxs_elite, idxs_rest, idxs_cat)
        nfit = optim_fn(npop)
        
        dfit = nfit[1:]-fit[idxs_rest]
        
        mri, mr = ucb.sample()
        ucb.report_reward(mri, -dfit.min())
        
        pop = npop
        fit = nfit
        data.append((pop, fit, mr))
    # pops, fits, mrs
    res = [torch.stack([d[i] for d in data]) for i in range(len(data[0]))]
    if return_ucb:
        return res, ucb
    else:
        return res

def run_evolution_ofmr(pop, optim_fn, n_gen, mrs, n_sample=1, 
                       k=.5, k_elite=None,
                       tqdm=lambda x: x):
    ofmr, f = calc_ofmr(pop, optim_fn, n_gen, mrs, n_sample, k=k, k_elite=k_elite)
    pops, fits_ofmr = run_evolution_base(pop, optim_fn, n_gen, mr=ofmr, 
                                               k=k, k_elite=None,
                                               tqdm=tqdm)
    return pops, fits_ofmr, ofmr.repeat(n_gen+1)
    