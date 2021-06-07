import os
import numpy as np
import torch
from torch import nn

import util
import models_decode
import population
import genotype

class Neuroevolution:
    def __init__(self, geno_config, evol_config, device='cpu', verbose=False):
        self.geno_config = geno_config
        self.evol_config = evol_config
        self.device = device
        
        self.best_fitness_ever_seen = None
        self.fitness_v_gen_AUC = 0
        
        pheno = self.geno_config['pheno_class'](**self.geno_config).to(self.device)
        pwl = len(util.model2vec(pheno))
        geno_config['pheno_weight_len'] = pwl
        decoder = self.geno_config['decoder_class'](**self.geno_config).to(self.device)
        breeder = self.geno_config['breeder_class'](**self.geno_config).to(self.device)
        dwl = len(util.model2vec(decoder))
        bwl = len(util.model2vec(breeder))
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
        if tqdm is not None:
            loop = tqdm(loop)
        for gen_idx in loop:
            self.pop = self.npop
            self.fitdata = self.calc_fitdata(self.pop)
            
            self.prob = population.fit2prob_sm(util.arr_dict2dict_arr(self.fitdata)['fitness'],
                                               **self.evol_config)
            self.npop = population.calc_next_population(self.pop, self.prob, 
                                                        self.calc_clone, self.calc_mutate, 
                                                        self.calc_crossover, **self.evol_config)
            loop.set_postfix({'fitness': np.max(util.arr_dict2dict_arr(self.fitdata)['fitness'])})
            
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
    
    
    