import os
import numpy as np
import torch
from torch import nn

import util
import models_decode
import population
import genotype

class CombinedGenotype():
    id_factory = 0
    def __init__(self, dna=None, decoder_dna=None, breeder_dna=None, 
                 genos_parent=None, orgin_type=None):
        """
        DNA is just another name for the genotype data of the 
        phenotype, breeder/crossover network, and the decoder.
        """
        self.dna = dna.clip(-2., 2.)
        self.decoder_dna = decoder_dna
        self.breeder_dna = breeder_dna
        
        if genos_parent is not None:
            self.parents_id = [geno.id for geno in genos_parent]
        
        self.id = CombinedGenotype.id_factory
        CombinedGenotype.id_factory += 1
    
    def generate_random(**kwargs):
        pc, dc, bc, device = [kwargs[i] for i in 
                              ['pheno_class', 'decoder_class', 'breeder_class', 'device']]
        
        # TODO this is not proper initialization of pheno_dna/dna
        if kwargs['dna_len']==kwargs['pheno_weight_len']:
            dna = util.model2vec(pc(**kwargs)).to(device)
        else:
            dna = kwargs['dna_init_std']*torch.randn(kwargs['dna_len']).to(device)
        decoder_dna = util.model2vec(dc(**kwargs)).to(device)
        breeder_dna = util.model2vec(bc(**kwargs)).to(device)
        return CombinedGenotype(dna, decoder_dna, breeder_dna, None, 'random')
        
    def clone(self):
        return CombinedGenotype(self.dna, self.decoder_dna, self.breeder_dna, [self], 'clone')
    
    def mutate(self, **kwargs):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        dna = self.dna
        if dna is not None:
            dna = util.perturb_type1(dna, kwargs['dna_mutate_lr']).detach()
            dna = util.perturb_type2(dna, kwargs['dna_mutate_prob']).detach()
        decoder_dna = self.decoder_dna
        if decoder_dna is not None:
            decoder_dna = util.perturb_type1(decoder_dna, kwargs['decode_mutate_lr']).detach()
            decoder_dna = util.perturb_type2(decoder_dna, kwargs['decode_mutate_prob']).detach()
        breeder_dna = self.breeder_dna
        if breeder_dna is not None:
            breeder_dna = util.perturb_type1(breeder_dna, kwargs['breed_mutate_lr']).detach()
            breeder_dna = util.perturb_type2(breeder_dna, kwargs['breed_mutate_prob']).detach()
        return CombinedGenotype(dna, decoder_dna, breeder_dna, [self], 'mutate')
    
    def crossover(self, geno2, breeder):
        breeder = util.vec2model(self.breeder_dna, breeder)
        dna = breeder.breed_dna(self.dna, geno2.dna)
        return CombinedGenotype(dna, self.decoder_dna, self.breeder_dna, [self, geno2], 'breed')
        
    def to_pheno(self, decoder, pheno):
        decoder = util.vec2model(self.decoder_dna, decoder)
        pheno_dna = decoder.decode_dna(self.dna)
        pheno = util.vec2model(pheno_dna, pheno)
        return pheno
    

class Neuroevolution:
    def __init__(self, geno_config, evol_config, device='cpu', verbose=False):
        self.geno_config = geno_config
        self.evol_config = evol_config
        self.device = device
        
        self.best_fitness_ever_seen = None
        self.fitness_v_gen_AUC = 0
        
        pheno, decoder, breeder = self.init_pheno_decoder_breeder()
        dwl = len(util.model2vec(decoder))
        bwl = len(util.model2vec(breeder))
        pwl = len(util.model2vec(pheno))
        geno_config['pheno_weight_len'] = pwl
        if geno_config['decoder_class'] is models_decode.IdentityDecoder:
            geno_config['dna_len'] = geno_config['pheno_weight_len']
        dna_len = geno_config['dna_len']
            
        if verbose:
            print('Running Neuroevolution with ')
            print('Population: ', evol_config['n_pop'])
            print('Generations: ', evol_config['n_gen'])
            print(f'Pheno   # params: {pwl:05d}')
            print('------------------')
            print(f'DNA     # params: {dna_len:05d}')
            print(f'Decoder # params: {dwl:05d}')
            print(f'Breeder # params: {bwl:05d}')
            print(f'Total   # params: {dna_len+dwl+bwl:5d}')
        
    def calc_clone(self, pop):
        return np.array([geno.clone() for geno in pop])

    def calc_mutate(self, pop):
        return np.array([geno.mutate(**self.geno_config) for geno in pop])
    
    def init_pheno_decoder_breeder(self):
        pheno = self.geno_config['pheno_class'](**self.geno_config).to(self.device)
        decoder = self.geno_config['decoder_class'](**self.geno_config).to(self.device)
        breeder = self.geno_config['breeder_class'](**self.geno_config).to(self.device)
        return pheno, decoder, breeder

    def calc_crossover(self, pop1, pop2):
        _, _, breeder = self.init_pheno_decoder_breeder()
        return np.array([geno1.crossover(geno2, breeder=breeder) for geno1, geno2 in zip(pop1, pop2)])
    
    def calc_fitdata(self, pop):
        fitdata = []
        pheno, decoder, _ = self.init_pheno_decoder_breeder()
        for geno in pop:
            pheno = geno.load_pheno(decoder, pheno)
            fitdata.append(self.evol_config['fitness_func'](pheno, device=self.device))
        return np.array(fitdata)
    
    def run_evolution(self, tqdm=None, logger=None, tag=None):
        self.npop = np.array([genotype.GenotypeCombined.generate_random(device=self.device, **self.geno_config) 
                              for _ in range(self.evol_config['n_pop'])])
        
        loop = range(self.evol_config['n_gen'])
        for gen_idx in (tqdm(loop) if tqdm is not None else loop):
            self.pop = self.npop
            self.fitdata = self.calc_fitdata(self.pop)
            
            self.prob = population.fit2prob_sm(util.arr_dict2dict_arr(self.fitdata)['fitness'],
                                               **self.evol_config)
            self.npop = population.calc_next_population(self.pop, self.prob, 
                                                        self.calc_clone, self.calc_mutate, 
                                                        self.calc_crossover, **self.evol_config)
            
            if logger is not None:
                self.log_stats(gen_idx, logger, tag)

    def log_stats(self, gen_idx, logger, tag=None):
        logger.add_scalar(f'{tag}/gpu_mem_allocated', torch.cuda.memory_allocated(), global_step=gen_idx)
        if gen_idx==0:
            self.fitdata_gens = []
            torch.save(self.geno_config, os.path.join(logger.log_dir, 'geno_config'))
        
#         fitdata = util.arr_dict2dict_arr(np.array([g.fitdata for g in self.pop_breeder]))
        fd_AD = self.fitdata
        fd_DA = util.arr_dict2dict_arr(fd_AD)
        self.fitdata_gens.append(fd_AD)
        for key, value in fd_DA.items():
            data = np.array(value)
            logger.add_histogram(f'{tag}/breeder_{key}', data, global_step=gen_idx)
#             data = data[data>=np.median(data)]
            logger.add_histogram(f'{tag}/breeder_{key}', data, global_step=gen_idx)
    
        d = {f'{tag}/best': np.max(fd_DA['fitness']), f'{tag}/worst': np.max(fd_DA['fitness'])}
        logger.add_scalars(f'fitnesses', d, global_step=gen_idx)
                
#         for key, value in self.fitdata.items():
#             data = np.array(value)
#             data = data[data>=np.median(data)]
#             logger.add_histogram(f'{tag}/{key}', data, global_step=gen_idx)
#         logger.add_histogram(f'{tag}/prob_of_selection', self.prob, global_step=gen_idx)
        
#         if self.pop[0].dna is not None:
#             all_dna_weights = torch.cat([geno.dna for geno in self.pop]).detach().cpu()
#             logger.add_histogram(f'{tag}/all_dna_weights', all_dna_weights, global_step=gen_idx)
#         if self.pop[0].decoder_dna is not None:
#             all_decoder_dna_weights = torch.cat([geno.decoder_dna for geno in self.pop]).detach().cpu()
#             if all_decoder_dna_weights.numel()>0:
#                 logger.add_histogram(f'{tag}/all_decoder_dna_weights', all_decoder_dna_weights, global_step=gen_idx)
#         if self.pop[0].breeder_dna is not None:
#             all_breeder_dna_weights = torch.cat([geno.breeder_dna for geno in self.pop]).detach().cpu()
#             if all_breeder_dna_weights.numel()>0:
#                 logger.add_histogram(f'{tag}/all_breeder_dna_weights', all_breeder_dna_weights, global_step=gen_idx)
        
#         torch.save(self.pop, os.path.join(logger.log_dir, f'pop_gen_{gen_idx:05d}.pop'))
#         torch.save(self.fitdata, os.path.join(logger.log_dir, f'fitdata_gen_{gen_idx:05d}'))
        
        
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
    
    
    