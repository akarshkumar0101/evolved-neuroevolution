import numpy as np
import torch
from torch import nn

import util

class Genotype():
    def __init__(self, dna=None, decoder_dna=None, breeder_dna=None, config=None):
        """
        DNA is just another name for the genotype data of the 
        phenotype, breeder/crossover network, and the decoder.
        """
        self.dna = dna
        self.decoder_dna = decoder_dna
        self.breeder_dna = breeder_dna
        self.config = config
    
    def generate_random(model, comodel, config, device='cpu'):
        dna = nn.utils.parameters_to_vector(model(config).parameters()).detach().to(device)
        decoder_dna = None
        breeder_dna = nn.utils.parameters_to_vector(comodel(config).parameters()).detach().to(device)
        return Genotype(dna, decoder_dna, breeder_dna, config)
        
    def clone(self):
        dna = None if self.dna is None else self.dna.clone()
        decoder_dna = None if self.decoder_dna is None else self.decoder_dna.clone()
        breeder_dna = None if self.breeder_dna is None else self.breeder_dna.clone()
        return Genotype(dna, decoder_dna, breeder_dna, self.config)
    
    def mutate(self):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        # mutate DNA
        dna = self.dna
        if dna is not None:
            dna = util.perturb_type1(dna, self.config['mutate_lr'])
            dna = util.perturb_type2(dna, self.config['mutate_prob'])

        # mutate decoder DNA
        decoder_dna = self.decoder_dna
        if decoder_dna is not None:
            decoder_dna = util.perturb_type1(decoder_dna, self.config['decode_mutate_lr'])
            decoder_dna = util.perturb_type2(decoder_dna, self.config['decode_mutate_prob'])
            
        # mutate breeder DNA
        breeder_dna = self.breeder_dna
        if breeder_dna is not None:
            breeder_dna = util.perturb_type1(breeder_dna, self.config['breed_mutate_lr'])
            breeder_dna = util.perturb_type2(breeder_dna, self.config['breed_mutate_prob'])
            
        return Genotype(dna, decoder_dna, breeder_dna, self.config)
        
    def crossover(self, another, conet):
        nn.utils.vector_to_parameters(self.breeder_dna, conet.parameters())
        dna = conet.crossover(self.dna, another.dna)
        return Genotype(dna, self.decoder_dna, self.breeder_dna, self.config)
    
    def to_pheno(self, pheno=None):
        nn.utils.vector_to_parameters(self.dna, pheno.parameters())
        return pheno

        

class CrossoverNE:
    def __init__(self, model, comodel, fitness_func, config, device='cpu'):
        self.model = model
        self.comodel = comodel
        self.fitness_func = fitness_func
        self.config = config
        
        self.device = device
        
        self.best_fitness = None
        self.fitness_v_gen_AUC = 0
        
    def get_init_population(self, N=None):
        if N is None:
            N = self.config['n_pop']
        return np.array([Genotype.generate_random(self.model, self.comodel, self.config, self.device) for _ in range(N)])

    def calc_fitnesses_and_sort(self):
        agent = self.model().to(self.device)
        self.fitdata = {}
        for geno in self.pop:
            agent = geno.to_pheno(agent)
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
        return np.array([geno.mutate() for geno in pop])
    
    def calc_crossover(self, pop1, pop2):
        conet = self.comodel(self.config).to(self.device)
        return np.array([geno1.crossover(geno2, conet) for geno1, geno2 in zip(pop1, pop2)])
        
    def fitnesses_to_prob(self, fitnesses, prob_sm_const=None, normalize=True):
        if prob_sm_const is None:
            prob_sm_const = self.config['prob_sm_const']
        prob = torch.from_numpy(fitnesses)
        if normalize:
            prob = prob/prob.std()
        prob = (prob_sm_const*prob).softmax(dim=-1).numpy()
        return prob
    
    def calc_next_population(self, pop, prob, crossover=False):
        """
        pop and fitnesses must be sorted by fitness.
        """
        npop = []
        npop.extend(pop[-self.config['n_elite']:])

        n_children = self.config['n_pop']-self.config['n_elite']
        if crossover:
            parents1 = np.random.choice(pop, size=n_children, p=prob, replace=self.config['with_replacement'])
            parents2 = np.random.choice(pop, size=n_children, p=prob, replace=self.config['with_replacement'])
            children = self.calc_crossover(parents1, parents2)
        else:
            children = np.random.choice(pop, size=n_children, p=prob, replace=self.config['with_replacement'])
        children = self.calc_mutate(children)
        npop.extend(children)
        return np.array(npop)
        
    def run_evolution(self, config=None, tqdm=None, logger=None, tag=None):
        if config is None:
            config = self.config 
            
        self.npop = self.get_init_population()
        
        loop = range(config['n_gen'])
        if tqdm is not None:
            loop = tqdm(loop)
            
        for gen_idx in loop:
            self.pop = self.npop
            self.calc_fitnesses_and_sort()
            self.prob = self.fitnesses_to_prob(self.fitnesses)
            self.npop = self.calc_next_population(self.pop, self.prob)

            if logger is not None:
                # logging
                self.log_stats(gen_idx, logger, tag)

    def log_stats(self, gen_idx, logger, tag=None):
        for key, value in self.fitdata.items():
            data = np.array(value)
            data = data[data>np.median(data)]
            logger.add_histogram(f'{tag}/{key}', data, global_step=gen_idx)
        logger.add_histogram(f'{tag}/prob', self.prob, global_step=gen_idx)
        
        self.best_fitness = np.max(self.fitnesses) if self.best_fitness is None else max(self.best_fitness, np.max(self.fitnesses))
                
        self.fitness_v_gen_AUC += np.max(self.fitnesses)
        
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
    
    
    