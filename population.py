import numpy as np
import torch

def fit2prob_sm(fitnesses, **kwargs):
    prob_sm_const =  kwargs['fit2prob_sm_iT']
    prob = torch.from_numpy(fitnesses)
    if kwargs['fit2prob_sm_normalize'] and prob.std().abs().item()>1e-3:
        prob = prob/prob.std()
    prob = (prob_sm_const*prob).softmax(dim=-1).numpy()
    return prob

def calc_next_population(pop, prob, calc_clone, calc_mutate, calc_crossover=None, **kwargs):
    npop = []

    n_elite_idxs = np.argsort(prob)[::-1][:kwargs['n_elite']]
    npop.extend(calc_clone(pop[n_elite_idxs]))

    n_children = len(pop)-len(npop)
    if calc_crossover is not None and kwargs['crossover']:
        parents1 = np.random.choice(pop, size=n_children, 
                                    p=prob, replace=kwargs['with_replacement'])
        parents2 = np.random.choice(pop, size=n_children,
                                    p=prob, replace=kwargs['with_replacement'])
        children = calc_crossover(parents1, parents2)
    else:
        children = np.random.choice(pop, size=n_children, 
                                    p=prob, replace=kwargs['with_replacement'])
    children = calc_mutate(children)
    npop.extend(children)
    return np.array(npop)

def calc_next_population2(pop, prob, calc_clone, calc_mutate, calc_crossover=None, n_elite=1, uniform_top_T=10):
    npop = []

    sort_idx = np.argsort(prob)[::-1]
    npop.extend(calc_clone(pop[sort_idx[:n_elite]]))

    n_children = len(pop)-len(npop)
    children = np.random.choice(pop[sort_idx[n_elite:n_elite+uniform_top_T]], size=n_children)
    children = calc_mutate(children)
    npop.extend(children)
    return np.array(npop)