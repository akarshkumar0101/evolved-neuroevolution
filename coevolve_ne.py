import numpy as np
import torch
from torch import nn

import util

class Genotype():
    def __init__(self, dna=None, config=None):
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
            dna = util.perturb_type1(dna, self.config['mutate_lr']).detach()
            dna = util.perturb_type2(dna, self.config['mutate_prob']).detach()

        # mutate decoder DNA
        decoder_dna = self.decoder_dna
        if decoder_dna is not None:
            decoder_dna = util.perturb_type1(decoder_dna, self.config['decode_mutate_lr']).detach()
            decoder_dna = util.perturb_type2(decoder_dna, self.config['decode_mutate_prob']).detach()
            
        # mutate breeder DNA
        breeder_dna = self.breeder_dna
        if breeder_dna is not None:
            breeder_dna = util.perturb_type1(breeder_dna, self.config['breed_mutate_lr']).detach()
            breeder_dna = util.perturb_type2(breeder_dna, self.config['breed_mutate_prob']).detach()
            
        return Genotype(dna, decoder_dna, breeder_dna, self.config)
        
    def crossover(self, another, conet):
        nn.utils.vector_to_parameters(self.breeder_dna, conet.parameters())
        dna = conet.crossover(self.dna, another.dna)
        return Genotype(dna, self.decoder_dna, self.breeder_dna, self.config)
    
    def to_pheno(self, pheno=None):
        nn.utils.vector_to_parameters(self.dna, pheno.parameters())
        return pheno
    
    
    
class Genotype():
    def __init__(self, dna=None, decoder_dna=None):
        self.dna = dna
        self.decoder_dna = decoder_dna
        
class CrossoverGenotype():
    def __init__(self, co_dna=None):
        self.co_dna = co_dna

class COCrossoverNE:
    def __init__(self, model, comodel, fitness_func, config, device='cpu'):
        self.model = model
        self.comodel = comodel
        self.fitness_func = fitness_func
        self.config = config
        
        self.device = device
        self.N = config['n_pop']
        self.co_N = config['n_pop']
        self.n_parents = self.N-self.config['n_elite']
        self.co_n_parents = self.co_N-self.config['n_elite']
        
        self.dim = len(nn.utils.parameters_to_vector(self.model().parameters()).detach())
        self.co_dim = len(nn.utils.parameters_to_vector(self.comodel().parameters()).detach())
        
    def geno2pheno(self, geno, pheno):
        nn.utils.vector_to_parameters(geno.dna, pheno.parameters())
        return pheno

    def set_init_population(self):
        pop, co_pop = [], []
        for i in range(self.N):
            dna = nn.utils.parameters_to_vector(self.model().parameters()).detach().to(self.device)
            pop.append(Genotype(dna, None))
        for i in range(self.co_N):
            co_dna = nn.utils.parameters_to_vector(self.comodel().parameters()).detach().to(self.device)
            co_pop.append(CrossoverGenotype(co_dna))
        self.npop = util.to_np_obj_array(pop)
        self.co_npop = util.to_np_obj_array(co_pop)

    def calc_fitnesses_of_pop(self, pop, sort=True):
        agent = self.model().to(self.device)
        fitdata = {}
        for geno in pop:
            agent = self.geno2pheno(geno, pheno=agent)
            fitdata_i = self.fitness_func(agent)
            for key in fitdata_i.keys():
                if key not in fitdata.keys():
                    fitdata[key] = []
                fitdata[key].append(fitdata_i[key])
        
        fit_idx = np.arange(len(pop))
        if sort:
            fit_idx = np.argsort(fitdata['fitness'])
            
        pop = pop[fit_idx]
        for key in fitdata.keys():
            fitdata[key] = np.array(fitdata[key])[fit_idx]
        fitnesses = fitdata['fitness']
        return pop, fitnesses, fitdata
        
    def calc_fitnesses_and_sort(self):
        self.pop, self.fitnesses, self.fitdata = self.calc_fitnesses_of_pop(self.pop, sort=True)
    
    def calc_co_fitnesses_and_sort(self):
        co_agent = self.comodel().to(self.device)
        self.co_fitnesses = []
        for co_geno in self.co_pop:
            nn.utils.vector_to_parameters(co_geno.co_dna, co_agent.parameters())
            
            n_children = 10
            parents1 = np.random.choice(self.pop, size=n_children, p=self.prob, replace=self.config['with_replacement'])
            parents2 = np.random.choice(self.pop, size=n_children, p=self.prob, replace=self.config['with_replacement'])
            co_pop_for_parents = util.to_np_obj_array([co_geno for _ in range(n_children)])
            children = self.calc_crossover(parents1, parents2, co_pop_for_parents)

            children, child_fitness, child_fitdata = self.calc_fitnesses_of_pop(children)
            co_fitness = child_fitness.mean()
            self.co_fitnesses.append(co_fitness)
        
        fit_idx = np.argsort(self.co_fitnesses)
        self.co_pop = self.co_pop[fit_idx]
        self.co_fitnesses = np.array(self.co_fitnesses)[fit_idx]
        
    def calc_mutate(self, pop):
        mutant = []
        for geno in pop:
            dna = geno.dna.clone()
            dna = dna + self.config['mutate_lr']*torch.randn_like(dna) # all small perturbation
            mutate_mask = torch.rand_like(dna)<self.config['mutate_prob']
            dna[mutate_mask] = 1e-1*torch.randn(mutate_mask.sum()).to(dna) # some extreme perturbations
            mutant.append(Genotype(dna))
        return mutant
    
    def calc_co_mutate(self, co_pop):
        mutant = []
        for geno in co_pop:
            co_dna = geno.co_dna.clone()
            co_dna = co_dna + self.config['co_mutate_lr']*torch.randn_like(co_dna) # all small perturbation
            mutate_mask = torch.rand_like(co_dna)<self.config['co_mutate_prob']
            co_dna[mutate_mask] = 1e-1*torch.randn(mutate_mask.sum()).to(co_dna) # some extreme perturbations
            mutant.append(CrossoverGenotype(co_dna))
        return mutant
    
    def calc_crossover(self, pop1, pop2, co_pop):
        cross = []
        conet = self.comodel().to(self.device)
        for geno1, geno2, co_geno in zip(pop1, pop2, co_pop):
            dna1, dna2, co_dna = geno1.dna, geno2.dna, co_geno.co_dna
            
            nn.utils.vector_to_parameters(co_dna, conet.parameters())
            dna = conet.crossover(geno1.dna, geno2.dna)
            
            cross.append(Genotype(dna, None))

        return util.to_np_obj_array(cross)
    
    def fitnesses_to_prob(self, fitnesses, prob_sm_const=None):
        if prob_sm_const is None:
            prob_sm_const = self.config['prob_sm_const']
        prob = torch.from_numpy(fitnesses)
        prob = (prob_sm_const*prob/prob.std()).softmax(dim=-1).numpy()
        return prob
        
    def calc_next_population(self):
        self.pop = self.npop
        self.co_pop = self.co_npop
        
        self.calc_fitnesses_and_sort()
        self.prob = self.fitnesses_to_prob(self.fitnesses)
        
        self.calc_co_fitnesses_and_sort()
        self.co_prob = self.fitnesses_to_prob(self.co_fitnesses)
        
        npop = []
        co_npop = []
        npop.extend(self.pop[-self.config['n_elite']:])
        co_npop.extend(self.co_pop[-self.config['co_n_elite']:])

        
        parents1 = np.random.choice(self.pop, size=self.n_parents, p=self.prob, replace=self.config['with_replacement'])
        parents2 = np.random.choice(self.pop, size=self.n_parents, p=self.prob, replace=self.config['with_replacement'])
        co_pop_for_parents = np.random.choice(self.co_pop, size=self.n_parents, p=self.co_prob, replace=self.config['with_replacement'])
        children = self.calc_crossover(parents1, parents2, co_pop_for_parents)
        children = self.calc_mutate(children)
        
        co_children = self.calc_co_mutate(co_pop_for_parents)

        npop.extend(children)
        co_npop.extend(co_children)

        self.npop = util.to_np_obj_array(npop)
        self.co_npop = util.to_np_obj_array(co_npop)

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



