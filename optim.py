import torch
import numpy as np
import cma

def calc_npop_truncate(pop, fit, k=.5, k_elite=None, mr=1e-2, mul_mr=False, idxs=None):
    k = int(len(pop)*k)
    if k_elite is None:
        k_elite = 1
    else:
        k_elite = int(len(pop)*k_elite)
    
    npop = torch.zeros_like(pop)
    idxs_sort = torch.argsort(fit)
    idxs_elite = idxs_sort[:k_elite]
    idxs_bum = idxs_sort[:k]
    
    npop[:k_elite] = pop[idxs_elite]
    
    n_children = len(pop)-k_elite
    
    if idxs is None:
        idxs = np.random.choice(len(idxs_bum), size=(n_children, ), replace=True)
        idxs = idxs_bum[idxs]
        
    if mul_mr:
        eps = -1+2*torch.rand(n_children, pop.shape[-1], 
                              device=pop.device, dtype=pop.dtype)
        eps = mr**eps
        children = pop[idxs]*eps
    else:
        children = pop[idxs]+mr*torch.randn_like(pop[idxs])
    npop[k_elite:] = children
    
    return npop, idxs

def calc_npop_ns(pop, mutpop, fit, k=.5, k_elite=None, mr_mut=2.):
    k = int(len(pop)*k)
    if k_elite is None:
        k_elite = 1
    else:
        k_elite = int(len(pop)*k_elite)
    
    npop = torch.zeros_like(pop)
    nmutpop = torch.zeros_like(mutpop)
    
    idxs_sort = torch.argsort(fit)
    idxs_elite = idxs_sort[:k_elite]
    idxs_bum = idxs_sort[:k]
    
    npop[:k_elite] = pop[idxs_elite]
    nmutpop[:k_elite] = mutpop[idxs_elite]
    
    n_children = len(pop)-k_elite
    
    idxs = np.random.choice(len(idxs_bum), size=(n_children, ), replace=True)
    idxs = idxs_bum[idxs]
    
    npop[k_elite:] = pop[idxs]+mutpop[idxs]*torch.randn_like(pop[idxs])
    
    eps = mr_mut**(-1+2*torch.rand_like(mutpop[idxs]))
    nmutpop[k_elite:] = mutpop[idxs]*eps
    return npop, nmutpop





def run_evolution_base(pop, optim_fn, n_gen, mr, tqdm=None):
    data = []
    loop = range(n_gen)
    if tqdm is not None: loop = tqdm(loop)
    for i in loop:
        fit = optim_fn(pop)
        data.append((pop, fit))
        pop, idxs = calc_npop_truncate(pop, fit, mr=mr)
    
    pops = torch.stack([d[0] for d in data])
    fits = torch.stack([d[1] for d in data])
    return pops, fits

def run_evolution_mutpops(pop, optim_fn, n_gen, n_mutpop=10, mr=None, mr_mut=2., tqdm=None):
    if mr is None:
        mutpop = torch.logspace(-3, 3, n_mutpop, device=pop.device)[:, None]
    else:
        mutpop = torch.linspace(mr, mr, n_mutpop, device=pop.device)[:, None]
        
    mut_assignment = np.arange(len(pop)-1)//int(len(pop)/len(mutpop))
    data = []
    loop = range(n_gen)
    if tqdm is not None: loop = tqdm(loop)
    for i in loop:
        fit = optim_fn(pop)
        
        mrs = mutpop[mut_assignment]
        fit_b = optim_fn(pop[1:])
        fit_a = optim_fn(pop[1:]+mrs*torch.randn_like(pop[1:]))
        fit_mrs = (fit_a-fit_b).reshape(len(mutpop), -1)
        data.append((pop, fit, mutpop, fit_mrs))
        
#         fit_mrs = fit_mrs.min(dim=-1).values
        fit_mrs = fit_mrs.sort(dim=-1).values[:, :1].mean(dim=-1)
        mutpop, _ = calc_npop_truncate(mutpop, fit_mrs, mr=mr_mut, mul_mr=True)
        pop, _ = calc_npop_truncate(pop, fit, mr=mrs)
        
    pops = torch.stack([d[0] for d in data])
    fits = torch.stack([d[1] for d in data])
    mutpops = torch.stack([d[2] for d in data])
    fitmrs = torch.stack([d[3] for d in data])
    return pops, fits, mutpops, fitmrs

def run_evolution_mutpops_full(pop, optim_fn, n_gen, n_mutpop=10, mr=None, mr_mut=2., tqdm=None):
    if mr is None:
        mutpop = torch.logspace(-3, 3, n_mutpop, device=pop.device)[:, None]
    else:
        mutpop = torch.linspace(mr, mr, n_mutpop, device=pop.device)[:, None]
        
    mut_assignment = np.arange(len(pop)-1)//int(len(pop)/len(mutpop))
    data = []
    
    fit = optim_fn(pop)
    
    loop = range(n_gen)
    if tqdm is not None: loop = tqdm(loop)
#     ffs = []
    for i in loop:
#         mutpop[-1,0] = 1e-1
        bpop = pop
        bfit = fit
        
        mrs = mutpop[mut_assignment]
        pop, idxs = calc_npop_truncate(pop, fit, mr=mrs)
        fit = optim_fn(pop)
        
        fit_mrs = (fit[1:]-bfit[idxs]).reshape(len(mutpop), -1)
        data.append((bpop, bfit, mutpop, fit_mrs))
#         ffs.append(fit_mrs)
        fit_mrs = fit_mrs.min(dim=-1).values
        # This is estimating the min of the (noisy) distribution by taking two std below the mean.
#         fit_mrs = fit_mrs.mean(dim=-1) - 2*fit_mrs.std(dim=-1)

#         if i%20==0:
#             fit_mrs = torch.cat(ffs, dim=-1)
#             fit_mrs = fit_mrs.min(dim=-1).values
        mutpop, _ = calc_npop_truncate(mutpop, fit_mrs, mr=mr_mut, mul_mr=True)
#             ffs = []
        
        
    pops = torch.stack([d[0] for d in data])
    fits = torch.stack([d[1] for d in data])
    mutpops = torch.stack([d[2] for d in data])
    fitmrs = torch.stack([d[3] for d in data])
    return pops, fits, mutpops, fitmrs

def run_evolution_mutpops_full_only_elite(pop, optim_fn, n_gen, n_mutpop=10, 
                                          mr=None, mr_mut=2., tqdm=None):
    if mr is None:
        mutpop = torch.logspace(-3, 3, n_mutpop, device=pop.device)[:, None]
    else:
        mutpop = torch.linspace(mr, mr, n_mutpop, device=pop.device)[:, None]
        
    mut_assignment = np.arange(len(pop)-1)//int(len(pop)/len(mutpop))
    data = []
    
    fit = optim_fn(pop)
    
    loop = range(n_gen)
    if tqdm is not None: loop = tqdm(loop)
#     ffs = []
    for i in loop:
        pop = pop[fit.argmin()].repeat(len(pop), 1)
        bpop = pop
        bfit = fit
        
        mrs = mutpop[mut_assignment]
        pop, idxs = calc_npop_truncate(pop, fit, mr=mrs)
        fit = optim_fn(pop)
        
        fit_mrs = (fit[1:]-bfit[idxs]).reshape(len(mutpop), -1)
        data.append((bpop, bfit, mutpop, fit_mrs))
#         ffs.append(fit_mrs)
        fit_mrs = fit_mrs.min(dim=-1).values
        # This is estimating the min of the (noisy) distribution by taking two std below the mean.
#         fit_mrs = fit_mrs.mean(dim=-1) - 2*fit_mrs.std(dim=-1)

#         if i%20==0:
#             fit_mrs = torch.cat(ffs, dim=-1)
#             fit_mrs = fit_mrs.min(dim=-1).values
        mutpop, _ = calc_npop_truncate(mutpop, fit_mrs, mr=mr_mut, mul_mr=True)
#             ffs = []
        
        
    pops = torch.stack([d[0] for d in data])
    fits = torch.stack([d[1] for d in data])
    mutpops = torch.stack([d[2] for d in data])
    fitmrs = torch.stack([d[3] for d in data])
    return pops, fits, mutpops, fitmrs

def run_evolution_ns(pop, optim_fn, n_gen, mr=None, mr_mut=2., tqdm=None):
    if mr is None:
        mutpop = torch.logspace(-3, 3, len(pop), device=pop.device)[:, None]
    else:
        mutpop = torch.linspace(mr, mr, len(pop), device=pop.device)[:, None]
    
    data = []
    loop = range(n_gen)
    if tqdm is not None: loop = tqdm(loop)
    for i in loop:
        fit = optim_fn(pop)
        pop, mutpop = calc_npop_ns(pop, mutpop, fit, mr_mut=2.)
#         pop, idxs = calc_npop_truncate(pop, fit, mr=mutpop)
#         mutpop, idxs = calc_npop_truncate(mutpop, fit[1:], mr=mr_mut, mul_mr=True, idxs=idxs)
        data.append((pop, fit, mutpop))
        
    pops = torch.stack([d[0] for d in data])
    fits = torch.stack([d[1] for d in data])
    mutpops = torch.stack([d[2] for d in data])
    return pops, fits, mutpops

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
    