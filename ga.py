import numpy as np
import torch

import util

def fit2prob_sm(fitnesses, **kwargs):
    temp = kwargs['temperature']
    prob = torch.from_numpy(fitnesses)
    if kwargs['normalize'] and prob.std().abs().item()>1e-3:
        prob = prob/prob.std()
    prob = (prob/temp).softmax(dim=-1).numpy()
    return prob

def calc_npop_roulette(pop, prob, calc_clone, calc_mutate, calc_crossover=None, **kwargs):
    k_elite = kwargs['k_elite']
    do_crossover = kwargs['do_crossover']
    with_replace = kwargs['with_replacement']
    
    npop = []
    idxs_sort = np.argsort(prob)
    idxs_elite = idxs_sort[-k_elite:]
    idxs_bum = idxs_sort[:-k_elite]
    npop.extend(calc_clone(pop[idxs_elite]))
    
    n_children = len(pop)-len(npop)
    
    pop, prob = pop[idxs_bum], prob[idxs_bum]
    prob = prob/prob.sum()

    if do_crossover:
        parents1 = np.random.choice(pop, size=n_children, 
                                    p=prob, replace=with_replace)
        parents2 = np.random.choice(pop, size=n_children,
                                    p=prob, replace=with_replace)
        children = calc_crossover(parents1, parents2)
    else:
        children = np.random.choice(pop, size=n_children, 
                                    p=prob, replace=with_replace)
    children = calc_mutate(children)
    npop.extend(children)
    return np.array(npop)

def calc_npop_tournament(pop, prob, calc_clone, calc_mutate, calc_crossover=None, **kwargs):
    k_elite = kwargs['k_elite']
    do_crossover = kwargs['do_crossover']
    with_replace = kwargs['with_replacement']
    
    k_tourn = kwargs['k_tournament']

    npop = []
    idxs_sort = np.argsort(prob)
    idxs_elite = idxs_sort[-k_elite:]
    idxs_bum = idxs_sort[:-k_elite]
    npop.extend(calc_clone(pop[idxs_elite]))
    
    n_children = len(pop)-len(npop)
    
    pop, prob = pop[idxs_bum], prob[idxs_bum]
    prob = prob/prob.sum()
    
    if do_crossover:
        parents1, parents2 = [], []
        for child_idx in range(n_children):
            idxs = np.random.randint(low=0, high=len(pop), size=(2, k_tourn))
            idxs_idxs_won = prob[idxs].argmax(axis=-1)
            idx1, idx2 = idxs[np.arange(2), idxs_idxs_won]
            parents1.append(pop[idx1])
            parents2.append(pop[idx2])
        parents1, parents2 = np.array(parents1), np.array(parents2)
        children = calc_crossover(parents1, parents2)
    else:
        for child_idx in range(n_children):
            idxs = np.random.randint(low=0, high=len(pop), size=(2, k_tourn))
            idxs_idxs_won = prob[idxs].argmax(axis=-1)
            idx1, idx2 = idxs[np.arange(2), idxs_idxs_won]
            parents1.append(pop[idx1])
        children = np.array(parents1)
    children = calc_mutate(children)
    npop.extend(children)
    return np.array(npop)
    
# TODO finish this method...
def calc_npop_top_K(pop, prob, calc_clone, calc_mutate, calc_crossover=None, **kwargs):
    k_elite, k_top = kwargs['select_k_elite'], kwargs['select_top_k']
    npop = []
    n_elite_idxs = np.argsort(prob)[::-1][:k_elite]
    npop.extend(calc_clone(pop[n_elite_idxs]))

    n_children = len(pop)-len(npop)
    children = np.random.choice(pop[sort_idx[k_elite:k_elite+k_top]], size=n_children)
    children = calc_mutate(children)
    npop.extend(children)
    return np.array(npop)

class SimpleGA:
    def __init__(self, calc_ipop=None, calc_clone=None, calc_mutate=None, calc_crossover=None, 
                 calc_npop=calc_npop_roulette, fit2prob=fit2prob_sm, fit2prob_cfg=None, select_cfg=None):
        self.gen_idx = 0
        
        self.calc_ipop = calc_ipop
        self.calc_clone = calc_clone
        self.calc_mutate = calc_mutate
        self.calc_crossover = calc_crossover
        
        self.calc_npop = calc_npop
        self.fit2prob = fit2prob_sm
        
        self.fit2prob_cfg = fit2prob_cfg
        self.select_cfg = select_cfg
    
    def ask(self):
        if self.gen_idx==0:
            self.npop = self.calc_ipop()
        self.pop = self.npop
        return self.pop
    
    def tell(self, fitdata):
        self.fitdata = fitdata
        for geno, fd in zip(self.pop, self.fitdata):
            geno.fitdata = fd
        self.fitdata_DA = util.arr_dict2dict_arr(self.fitdata)
        self.prob = self.fit2prob(self.fitdata_DA['fitness'], **self.fit2prob_cfg)
        self.npop = self.calc_npop(self.pop, self.prob, self.calc_clone, self.calc_mutate, 
                                   self.calc_crossover, **self.select_cfg)
        self.gen_idx += 1
        
    def run_evolution(self, n_gens, calc_fitdata, tqdm=None, fn_callback=None):
        loop = range(n_gens)
        if tqdm is not None:
            loop = tqdm(loop)
            
        for gen_idx in loop:
            fitdata = calc_fitdata(self.ask())
            
            if False:
                data = torch.stack([g.geno_dna.dna for g in self.pop])
                data = util.calc_pairwise_corr(data).mean(dim=0).detach().cpu().numpy()
                for fd, sim in zip(fitdata, data):
                    fd['fitness'] -= sim/10.

            self.tell(fitdata)
            
            if tqdm is not None:
                loop.set_postfix({'fitness': np.max(self.fitdata_DA['fitness'])})
            if fn_callback is not None:
                fn_callback()
        