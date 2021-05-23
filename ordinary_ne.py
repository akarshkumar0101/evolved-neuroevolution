import numpy as np
import torch
from torch import nn

class OrdinaryNE:
    def __init__(self, model, fitness_func, config, device='cpu'):
        self.model = model
        self.fitness_func = fitness_func
        self.config = config
        
        self.device = device
        self.N = config['n_pop']
        
    def geno2pheno(self, geno, pheno):
        nn.utils.vector_to_parameters(geno, pheno.parameters())
        return pheno

    def set_init_population(self):
        pop = np.empty(self.N, dtype=object)
        for i in range(self.N):
            geno = nn.utils.parameters_to_vector(self.model().parameters()).detach().to(self.device)
            pop[i] = geno
        self.npop = pop

    def calc_fitnesses_and_sort(self):
        agent = self.model().to(self.device)
        self.fitdata = {}
        for geno in self.pop:
            agent = self.geno2pheno(geno, pheno=agent)
            fitdata_i = self.fitness_func(agent)
            for key in fitdata_i.keys():
                if key not in self.fitdata.keys():
                    self.fitdata[key] = []
                self.fitdata[key].append(fitdata_i[key])
        
        fit_idx = np.argsort(self.fitdata['fitness'])
        self.pop = self.pop[fit_idx]
        for key in self.fitdata.keys():
            self.fitdata[key] = np.array(self.fitdata[key])[fit_idx]
        self.fitnesses = self.fitdata['fitness']

    def calc_mutate(self, pop):
        mutant = []
        for p in pop:
            pm = p.detach().clone()
            pm = pm + self.config['lr']*torch.randn_like(pm) # all small perturbation
            mutate_mask = torch.rand_like(pm)<self.config['prob']
            pm[mutate_mask] = torch.randn(mutate_mask.sum()).to(pm) # some extreme perturbations
            mutant.append(pm)
        return mutant

    def calc_crossover(self, pop1, pop2):
        cross = []
        for geno1, geno2 in zip(pop1, pop2):
            geno = geno1.clone()

            # with half and half
    #         mask = torch.arange(len(geno1))<(len(geno1)//2)
    #         geno = geno2.clone()
    #         geno[mask] = geno1[mask]

            # with mean
    #         geno = torch.stack([geno1, geno2], dim=0).mean(dim=0)


            cross.append(geno)

        return cross
    
        

    def calc_next_population(self):
        self.pop = self.npop
        self.calc_fitnesses_and_sort()
        
        npop = []
        npop.extend(self.pop[-self.config['n_elite']:])

        self.prob = torch.from_numpy(self.fitnesses)
        self.prob = (self.config['prob_sm_const']*self.prob/self.prob.std()).softmax(dim=-1).numpy()
        
        n_parents = self.N-self.config['n_elite']
        parents1 = np.random.choice(self.pop, size=n_parents, p=self.prob, replace=self.config['with_replacement'])
        parents2 = np.random.choice(self.pop, size=n_parents, p=self.prob, replace=self.config['with_replacement'])
        children = self.calc_crossover(parents1, parents2)
        children = self.calc_mutate(children)

        npop.extend(children)

        npop_arr = np.empty(self.N, dtype=object)
        for i in range(self.N):
            npop_arr[i] = npop[i]

        self.npop = npop_arr

    def run_evolution(self, config=None, tqdm=None, logger=None, tag=None):
        if config is None:
            config = self.config 
            
        self.set_init_population()
        
        loop = range(config['n_gen'])
        if tqdm is not None:
            loop = tqdm(loop)
            
        for gen_idx in loop:
            self.calc_next_population()

            if logger is not None:
                # logging
                for key, value in self.fitdata.items():
                    data = np.array(value)
                    data = data[data>np.median(data)]
                    logger.add_histogram(f'{tag}/{key}', data, global_step=gen_idx)
                logger.add_histogram(f'{tag}/prob', self.prob, global_step=gen_idx)


        #         logger.add_scalars(f'bigger both perturb lr={lr}, prob={prob}',
        #                            {'max': np.max(fitnesses['nll']),
        #                             'median': np.median(fitnesses['nll']),
        #                             'min': np.min(fitnesses['nll']),
        #                            }, global_step=gen_idx)
                if gen_idx%1==0:
                    pass
        #             plt.bar(np.arange(len(prob)), np.sort(np.array(prob)))
        #             plt.show()
#                     a = torch.stack(list(population))
#                     a = (a[:, None, :]-a[None, :, :]).norm(dim=-1).flatten()
#                     logger.add_histogram('tag_similar', a, global_step=gen_idx)
            #         plt.imshow()
            #         plt.colorbar()
            #         logger.add_figure('similarity', plt.gcf(), global_step=gen_idx)
        return self.pop



