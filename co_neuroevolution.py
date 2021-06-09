import os
import numpy as np
import torch
from torch import nn

import util
import models_decode
import ga
import genotype

class Neuroevolution:
    def __init__(self, geno_cfg, evol_cfg, device='cpu', verbose=False, logger=None, tag=None):
        self.logger = logger
        self.tag = tag
        
        self.geno_cfg = geno_cfg
        self.evol_cfg = evol_cfg
        self.device = device
        
        f2pcfg = self.evol_cfg['fit2prob_cfg']
        self.ga = ga.SimpleGA(self.calc_ipop, self.calc_clone, self.calc_mutate, self.calc_crossover,
                              fit2prob_cfg=f2pcfg, select_cfg=self.evol_cfg['select_cfg'])

        self.ga_decoder = ga.SimpleGA(self.calc_ipop_decoder, self.calc_clone, self.calc_mutate, None,
                                      fit2prob_cfg=f2pcfg, select_cfg=self.evol_cfg['select_cfg_decoder'])
        self.ga_breeder = ga.SimpleGA(self.calc_ipop_breeder, self.calc_clone, self.calc_mutate, None,
                                      fit2prob_cfg=f2pcfg, select_cfg=self.evol_cfg['select_cfg_breeder'])
        
        
        self.pheno = self.geno_cfg['pheno_class'](**self.geno_cfg).to(self.device)
        pwl = len(util.model2vec(self.pheno))
        geno_cfg['pheno_weight_len'] = pwl
        if geno_cfg['decoder_class'] is models_decode.IdentityDecoder:
            geno_cfg['dna_len'] = geno_cfg['pheno_weight_len']
        dna_len = geno_cfg['dna_len']
        self.decoder = self.geno_cfg['decoder_class'](**self.geno_cfg).to(self.device)
        dwl = len(util.model2vec(self.decoder))
        self.breeder = self.geno_cfg['breeder_class'](**self.geno_cfg).to(self.device)
        bwl = len(util.model2vec(self.breeder))
            
        if verbose:
            print('Running Neuroevolution with ')
            print('Population: ', evol_cfg['n_pop'])
            print('Generations: ', evol_cfg['n_gen'])
            print(f'Pheno   # params: {pwl:05d}')
            print('------------------')
            print(f'DNA     # params: {dna_len:05d}')
            print(f'Decoder # params: {dwl:05d}')
            print(f'Breeder # params: {bwl:05d}')
            print(f'Total   # params: {dna_len+dwl+bwl:5d}')
        
    def calc_ipop(self):
        return np.array([genotype.GenotypeContinuousDNA.generate_random(device=self.device, **self.geno_cfg)
                         for _ in range(self.evol_cfg['n_pop'])])
    
    def calc_ipop_decoder(self):
        return np.array([genotype.GenotypeDecoderWeights.generate_random(device=self.device, **self.geno_cfg)
                         for _ in range(self.evol_cfg['n_pop_decoder'])])
    
    def calc_ipop_breeder(self):
        return np.array([genotype.GenotypeBreederWeights.generate_random(device=self.device, **self.geno_cfg)
                         for _ in range(self.evol_cfg['n_pop_breeder'])])
        
    def calc_clone(self, pop):
        return np.array([geno.clone() for geno in pop])

    def calc_mutate(self, pop):
        return np.array([geno.mutate(**self.geno_cfg) for geno in pop])
    
    def calc_crossover(self, pop1, pop2):
        # TODO make this sample from breeders based on breeder fitness
        geno_breeders = np.random.choice(self.ga_breeder.pop, size=len(pop1))
        return np.array([geno1.crossover(geno2, geno_breeder, breeder=self.breeder) 
                         for geno_breeder, geno1, geno2 in zip(geno_breeders, pop1, pop2)])

    def calc_fitdata(self, pop, n_sample=None):
        if n_sample is None:
            n_sample = self.evol_cfg['n_sample_eval_dna']
        if n_sample==0:
            return np.array([{'fitness': 0.} for _ in pop])
        fitdata = []
        for geno in pop:
            for geno_decoder in np.random.choice(self.ga_decoder.pop, size=n_sample, replace=False):
                pheno = geno_decoder.load_pheno(geno, self.decoder, self.pheno)
                fitdata.append(self.evol_cfg['fitness_func'](pheno, device=self.device))
        return util.arr_dict_mean(np.array(fitdata).reshape((len(pop), n_sample)), axis=-1) 
        
    def calc_fitdata_decoder(self, pop_decoder, n_sample=None):
        if n_sample is None:
            n_sample = self.evol_cfg['n_sample_eval_decoder']
        if n_sample==0:
            return np.array([{'fitness': 0.} for _ in pop_decoder])
        fitdata = []
        for geno_decoder in pop_decoder:
            for geno in np.random.choice(self.ga.pop, size=n_sample, replace=False):
                pheno = geno_decoder.load_pheno(geno, self.decoder, self.pheno)
                fitdata.append(self.evol_cfg['fitness_func'](pheno, device=self.device))
        return util.arr_dict_mean(np.array(fitdata).reshape((len(pop_decoder), n_sample)), axis=-1) 
    
    def calc_fitdata_breeder(self, pop_breeder, n_sample=None):
        if n_sample is None:
            n_sample = self.evol_cfg['n_sample_eval_breeder']
        if n_sample==0:
            return np.array([{'fitness': 0.} for _ in pop_breeder])
        fitdata = []
        for geno_breeder in pop_breeder:
            for geno1, geno2 in np.random.choice(self.ga.pop, size=(n_sample, 2), replace=True):
                geno_decoder = np.random.choice(self.ga_decoder.pop)
                geno = geno1.crossover(geno2, geno_breeder, self.breeder)
                pheno = geno_decoder.load_pheno(geno, self.decoder, self.pheno)
                fd = self.evol_cfg['fitness_func'](pheno, device=self.device)
                fd = {key: fd[key]-(geno1.fitdata[key]+geno2.fitdata[key])/2. for key in fd.keys()}
                fitdata.append(fd)
        return util.arr_dict_mean(np.array(fitdata).reshape((len(pop_breeder), n_sample)), axis=-1) 
    
    def run_evolution(self, tqdm=None):
        loop = range(self.evol_cfg['n_gen'])
        if tqdm is not None:
            loop = tqdm(loop)
            
        for gen_idx in loop:
            pop, pop_d, pop_b = self.ga.ask(), self.ga_decoder.ask(), self.ga_breeder.ask()
            fitdata = self.calc_fitdata(pop)
            self.ga.tell(fitdata)
            fitdata = self.calc_fitdata_decoder(pop_d)
            self.ga_decoder.tell(fitdata)
            fitdata = self.calc_fitdata_breeder(pop_b)
            self.ga_breeder.tell(fitdata)
            
            if tqdm is not None:
                loop.set_postfix({'fitness': np.max(self.ga.fitdata_DA['fitness'])})
            self.log_stats()
            
    def log_stats(self, ):
        gen_idx = self.ga.gen_idx
        logger = self.logger
        tag = self.tag
        if logger is None:
            return
        
        logger.add_scalar(f'{tag}/gpu_mem_allocated', torch.cuda.memory_allocated(), global_step=gen_idx)
        if gen_idx==1:
            self.fitdata_gens = []
            torch.save(self.geno_cfg, os.path.join(logger.log_dir, 'geno_cfg'))
        
#         fitdata = util.arr_dict2dict_arr(np.array([g.fitdata for g in self.pop_breeder]))
        fd_AD = self.ga.fitdata
        fd_DA = util.arr_dict2dict_arr(fd_AD)
        self.fitdata_gens.append(fd_AD)
        for key, value in fd_DA.items():
            data = np.array(value)
#             logger.add_histogram(f'{tag}/breeder_{key}', data, global_step=gen_idx)
#             data = data[data>=np.median(data)]
#             logger.add_histogram(f'{tag}/breeder_{key}', data, global_step=gen_idx)
    
#         d = {f'{tag}/best': np.max(fd_DA['fitness']), f'{tag}/worst': np.max(fd_DA['fitness'])}
#         logger.add_scalars(f'fitnesses', d, global_step=gen_idx)
                
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
    
    
    
    
    