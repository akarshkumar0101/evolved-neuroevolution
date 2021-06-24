import numpy as np
import torch
import cma

import util

def run_cmaes(model, fitness_func, n_gen, n_pop, device=None, tqdm=None):
    net = model().to(device)

    def fitness_wrapper(sol, net=net):
        sol = torch.tensor(sol, device=device).to(torch.float32)
        net = util.vec2model(sol, net).to(device)
        fitdata = fitness_func(net, device=device)
        return fitdata

    sol = util.model2vec(net).detach().cpu().numpy()
    es = cma.CMAEvolutionStrategy(sol, 1e-1, {'popsize':n_pop})
    # es.optimize(pheno_fitness)
    fitdata_gens = []
    loop = range(n_gen)
    if tqdm is not None:
        loop = tqdm(loop)
    for _ in loop:
        sols = es.ask()
        fitdata = np.array([fitness_wrapper(sol) for sol in sols])
        fitdata_DA = util.arr_dict2dict_arr(np.array([fitness_wrapper(sol) for sol in sols]))
        es.tell(sols, fitdata_DA['loss'])
        fitdata_gens.append(fitdata)
        if tqdm is not None:
            loop.set_postfix({'loss': fitdata_DA['loss'].min()})
            
    return np.array(fitdata_gens)