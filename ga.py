import numpy as np
import torch

from functools import partial

import util

def fit2prob_sm(fitnesses, **kwargs):
    temp = kwargs['temperature']
    normalize = kwargs['normalize']
    prob = torch.from_numpy(fitnesses)
    if normalize and prob.std().abs().item()>1e-3:
        prob = prob/prob.std()
    prob = (prob/temp).softmax(dim=-1).numpy()
    return prob

def fit2prob_top_K(fitnesses, **kwargs):
    k = kwargs['top_k']
    prob = np.zeros_like(fitnesses)
    prob[fitnesses.argsort()[-k:]] = 1.
    prob = prob/prob.sum()
    return prob

def calc_npop_roulette(pop, fitnesses, **kwargs):
    k_elite = kwargs['k_elite']
    do_crossover = kwargs['do_crossover']
    
    calc_clone = kwargs['calc_clone_fn']
    calc_mutate = kwargs['calc_mutate_fn']
    calc_crossover = kwargs['calc_crossover_fn']
    
    fit2prob = kwargs['fit2prob_fn']
    
    prob = fit2prob(fitnesses)
    
    npop = []
    idxs_sort = np.argsort(fitnesses)[::-1]
    idxs_elite = idxs_sort[:k_elite]
    idxs_bum = idxs_sort[k_elite:]
    npop.extend(calc_clone(pop[idxs_elite]))
    
    n_children = len(pop)-len(npop)
    
    pop, prob = pop[idxs_bum], prob[idxs_bum]
    prob = prob/prob.sum()

    parents1, parents2 = np.random.choice(pop, size=(2, n_children), p=prob)
    if do_crossover:
        children = calc_crossover(parents1, parents2)
    else:
        children = parents1
    children = calc_mutate(children)
    npop.extend(children)
    return util.to_np_obj_array(npop)

def calc_npop_tournament(pop, fitnesses, **kwargs):
    k_elite = kwargs['k_elite']
    do_crossover = kwargs['do_crossover']
    
    k_tourn = kwargs['k_tournament']
    
    calc_clone = kwargs['calc_clone_fn']
    calc_mutate = kwargs['calc_mutate_fn']
    calc_crossover = kwargs['calc_crossover_fn']

    npop = []
    idxs_sort = np.argsort(fitnesses)[::-1]
    idxs_elite = idxs_sort[:k_elite]
    idxs_bum = idxs_sort[k_elite:]
    npop.extend(calc_clone(pop[idxs_elite]))
    
    n_children = len(pop)-len(npop)
    
    pop, fitnesses = pop[idxs_bum], fitnesses[idxs_bum]
    
    parents1, parents2 = [], []
    for child_idx in range(n_children):
        idxs = np.random.randint(low=0, high=len(pop), size=(2, k_tourn)).min(axis=-1)
        p1, p2 = pop[idxs]
        parents1.append(p1)
        parents2.append(p2)
    parents1, parents2 = util.to_np_obj_array(parents1), util.to_np_obj_array(parents2)
    if do_crossover:
        children = calc_crossover(parents1, parents2)
    else:
        children = parents1
    children = calc_mutate(children)
    npop.extend(children)
    return util.to_np_obj_array(npop)

    
    
class SimpleGA:
    def __init__(self, **kwargs):
        self.gen_idx = 0
        
        self.calc_ipop = kwargs['calc_ipop_fn']
        self.calc_npop = kwargs['calc_npop_fn']
        
        self.calc_npop = partial(self.calc_npop,
                                 calc_clone_fn=kwargs['calc_clone_fn'],
                                 calc_mutate_fn=kwargs['calc_mutate_fn'],
                                 calc_crossover_fn=kwargs['calc_crossover_fn'])
    
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
        self.fitness = self.fitdata_DA['fitness']
        self.npop = self.calc_npop(self.pop, self.fitness)
        self.gen_idx += 1
        
    def run_evolution(self, n_gens, calc_fitdata_fn, tqdm=None, fn_callback=None):
        loop = range(n_gens)
        if tqdm is not None:
            loop = tqdm(loop)
            
        for gen_idx in loop:
            fitdata = calc_fitdata_fn(self.ask())
            
            self.tell(fitdata)
            
            if tqdm is not None:
                best_agent_fitdata = self.fitdata[np.argmax(self.fitdata_DA['fitness'])]
                loop.set_postfix(best_agent_fitdata)
            if fn_callback is not None:
                fn_callback(self)
        