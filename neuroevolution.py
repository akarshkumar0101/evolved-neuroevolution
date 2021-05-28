import os
import numpy as np
import torch
from torch import nn

import util
import models_decode

class Genotype():
    def __init__(self, dna=None, decoder_dna=None, breeder_dna=None, config=None):
        """
        DNA is just another name for the genotype data of the 
        phenotype, breeder/crossover network, and the decoder.
        """
        self.dna = dna.clip(-2., 2.)
        self.decoder_dna = decoder_dna
        self.breeder_dna = breeder_dna
        self.config = config
    
    def generate_random(model_pheno, model_decode, model_breed, config, device='cpu'):
        # TODO this is not proper initialization of pheno_dna/dna
        if config['dna_len']==config['pheno_weight_len']:
            dna = model_pheno.generate_random(config, device)
        else:
            dna = config['dna_init_std']*torch.randn(config['dna_len']).to(device)
        decoder_dna = model_decode.generate_random(config, device)
        breeder_dna = model_breed.generate_random(config, device)
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
        
    def crossover(self, geno2, breeder):
        breeder.load_breeder_dna(self.breeder_dna)
        dna = breeder.breed_dna(self.dna, geno2.dna)
        return Genotype(dna, self.decoder_dna, self.breeder_dna, self.config)
    
    def to_pheno(self, decoder=None, pheno=None):
        decoder.load_decoder_dna(self.decoder_dna)
        pheno_dna = decoder.decode_dna(self.dna)
        pheno.load_pheno_dna(pheno_dna)
        return pheno


class Neuroevolution:
    def __init__(self, model_pheno, model_decode, model_breed, fitness_func, config, device='cpu', verbose=False):
        self.model_pheno = model_pheno
        self.model_decode = model_decode
        self.model_breed = model_breed
        self.fitness_func = fitness_func
        self.config = config
        
        self.device = device
        
        self.best_fitness = None
        self.fitness_v_gen_AUC = 0
        
        self.config['pheno_weight_len'] = len(nn.utils.parameters_to_vector(model_pheno(config).parameters()).detach())
        if model_decode is models_decode.IdentityDecoder:
            self.config['dna_len'] = self.config['pheno_weight_len']
            
        if verbose:
            print('Running Neuroevolution with ')
            print('DNA Length: ', self.config['dna_len'])
        
        
    def get_init_population(self, N=None):
        if N is None:
            N = self.config['n_pop']
        pop = np.array([Genotype.generate_random(self.model_pheno, 
                                                 self.model_decode, self.model_breed, 
                                                 self.config, self.device) for _ in range(N)])
        parent_idxs = np.arange(N)
        return pop, parent_idxs

    def calc_fitnesses_and_sort(self):
        decoder = self.model_decode(self.config).to(self.device)
        agent = self.model_pheno(self.config).to(self.device)
        self.fitdata = {}
        for geno in self.pop:
            agent = geno.to_pheno(decoder, agent)
            fitdata_i = self.fitness_func(agent)
            for key in fitdata_i.keys():
                if key not in self.fitdata.keys():
                    self.fitdata[key] = []
                self.fitdata[key].append(fitdata_i[key])
        
        fit_idx = np.argsort(self.fitdata['fitness'])[::-1]
        self.pop = self.pop[fit_idx]
        self.parent_idxs = [self.parent_idxs[i] for i in fit_idx]
        for key in self.fitdata.keys():
            self.fitdata[key] = np.array(self.fitdata[key])[fit_idx]
            
        self.fitnesses = self.fitdata['fitness']

    def calc_mutate(self, pop):
        return np.array([geno.mutate() for geno in pop])
    
    def calc_crossover(self, pop1, pop2):
        breeder = self.model_breed(self.config).to(self.device)
        return np.array([geno1.crossover(geno2, breeder) for geno1, geno2 in zip(pop1, pop2)])
        
    def fitnesses_to_prob(self, fitnesses, prob_sm_const=None, normalize=True):
        if prob_sm_const is None:
            prob_sm_const = self.config['prob_sm_const']
        prob = torch.from_numpy(fitnesses)
        if normalize and prob.std().abs().item()>1e-3:
            prob = prob/prob.std()
        prob = (prob_sm_const*prob).softmax(dim=-1).numpy()
        return prob
    
    def calc_next_population(self, pop, prob, crossover=True):
        """
        pop and fitnesses must be sorted by fitness.
        """
        npop = []
        parent_idxs = []
        
        npop.extend(pop[:self.config['n_elite']])
        parent_idxs.extend(np.arange(self.config['n_elite']))
        
        n_children = self.config['n_pop']-self.config['n_elite']
        if crossover:
            parents1_idx = np.random.choice(np.arange(len(pop)), size=n_children, 
                                            p=prob, replace=self.config['with_replacement'])
            parents2_idx = np.random.choice(np.arange(len(pop)), size=n_children, 
                                            p=prob, replace=self.config['with_replacement'])
            parents1 = pop[parents1_idx]
            parents2 = pop[parents2_idx]
            children = self.calc_crossover(parents1, parents2)
            
            parent_idxs.extend(np.stack([parents1_idx, parents2_idx], axis=-1))
        else:
            parents1_idx = np.random.choice(np.arange(len(pop)), size=n_children, 
                                            p=prob, replace=self.config['with_replacement'])
            children = pop[parents1_idx]
            parent_idxs.extend(parents1_idx)
        children = self.calc_mutate(children)
        npop.extend(children)
        return np.array(npop), parent_idxs
        
    def run_evolution(self, config=None, tqdm=None, logger=None, tag=None):
        if config is None:
            config = self.config
            
        self.npop, self.nparent_idxs = self.get_init_population()
        
        loop = range(config['n_gen'])
            
        for gen_idx in (tqdm(loop) if tqdm is not None else loop):
            self.pop, self.parent_idxs = self.npop, self.nparent_idxs
            self.calc_fitnesses_and_sort()
            self.prob = self.fitnesses_to_prob(self.fitnesses)
            self.npop, self.nparent_idxs = self.calc_next_population(self.pop, self.prob)

            if logger is not None:
                # logging
                self.log_stats(gen_idx, logger, tag)

    def log_stats(self, gen_idx, logger, tag=None):
        self.best_fitness = np.max(self.fitnesses) if self.best_fitness is None else max(self.best_fitness, np.max(self.fitnesses))
                
        self.fitness_v_gen_AUC += np.max(self.fitnesses)
        
        for key, value in self.fitdata.items():
            data = np.array(value)
            data = data[data>=np.median(data)]
            logger.add_histogram(f'{tag}/{key}', data, global_step=gen_idx)
        logger.add_histogram(f'{tag}/prob_of_selection', self.prob, global_step=gen_idx)
        
        if self.pop[0].dna is not None:
            all_dna_weights = torch.cat([geno.dna for geno in self.pop]).detach().cpu().numpy()
            logger.add_histogram(f'{tag}/all_dna_weights', all_dna_weights, global_step=gen_idx)
        if self.pop[0].decoder_dna is not None:
            all_decoder_dna_weights = torch.cat([geno.decoder_dna for geno in self.pop]).detach().cpu().numpy()
            logger.add_histogram(f'{tag}/all_decoder_dna_weights', all_decoder_dna_weights, global_step=gen_idx)
        if self.pop[0].breeder_dna is not None:
            all_breeder_dna_weights = torch.cat([geno.breeder_dna for geno in self.pop]).detach().cpu().numpy()
            logger.add_histogram(f'{tag}/all_breeder_dna_weights', all_breeder_dna_weights, global_step=gen_idx)
        
        
        logger.add_scalar(f'{tag}/gpu_mem_allocated', torch.cuda.memory_allocated(), global_step=gen_idx)
        
        torch.save(self.pop, os.path.join(logger.log_dir, f'pop_gen_{gen_idx:05d}'))
        torch.save(self.fitnesses, os.path.join(logger.log_dir, f'fitnesses_gen_{gen_idx:05d}'))
        torch.save(self.parent_idxs, os.path.join(logger.log_dir, f'parent_idxs_gen_{gen_idx:05d}'))
        
        if gen_idx==0:
            torch.save(self.config, os.path.join(logger.log_dir, 'config'))
        
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
    
    
    