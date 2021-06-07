import os
import numpy as np
import torch
from torch import nn

import util
import models_decode
import population

class DNAGenotype():
    id_factory = 0
    def __init__(self, dna=None, genos_parent=None, orgin_type=None):
        self.dna = dna.clip(-2., 2.)
        
        if genos_parent is not None:
            self.parents_id = [geno.id for geno in genos_parent]
        
        self.id = DNAGenotype.id_factory
        DNAGenotype.id_factory += 1
    
    def generate_random(**kwargs):
        pc, device = [kwargs[i] for i in ['pheno_class', 'device']]
        
        # TODO this is not proper initialization of pheno_dna/dna
        if kwargs['dna_len']==kwargs['pheno_weight_len']:
            dna = util.model2vec(pc(**kwargs)).to(device)
        else:
            dna = kwargs['dna_init_std']*torch.randn(kwargs['dna_len']).to(device)
        return DNAGenotype(dna, None, 'random')
        
    def clone(self):
        return DNAGenotype(self.dna, [self], 'clone')
    
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
        return DNAGenotype(dna, [self], 'mutate')
    
    def crossover(self, geno_breeder, geno2, breeder):
        breeder = util.vec2model(geno_breeder.breeder_dna, breeder)
        dna = breeder.breed_dna(self.dna, geno2.dna)
        return DNAGenotype(dna, [self, geno2], 'breed')
        
    def to_pheno(self, geno_decoder, decoder, pheno):
        decoder = util.vec2model(geno_decoder.decoder_dna, decoder)
        pheno_dna = decoder.decode_dna(self.dna)
        pheno = util.vec2model(pheno_dna, pheno)
        return pheno
    
class DecoderGenotype():
    id_factory = 0
    def __init__(self, decoder_dna=None, genos_parent=None, orgin_type=None):
        self.decoder_dna = decoder_dna
        
        if genos_parent is not None:
            self.parents_id = [geno.id for geno in genos_parent]
        
        self.id = DecoderGenotype.id_factory
        DecoderGenotype.id_factory += 1
    
    def generate_random(**kwargs):
        dc, device = [kwargs[i] for i in ['decoder_class', 'device']]
        
        decoder_dna = util.model2vec(dc(**kwargs)).to(device)
        return DecoderGenotype(decoder_dna, None, 'random')
        
    def clone(self):
        return DecoderGenotype(self.decoder_dna, [self], 'clone')
    
    def mutate(self, **kwargs):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        decoder_dna = self.decoder_dna
        if decoder_dna is not None:
            decoder_dna = util.perturb_type1(decoder_dna, kwargs['decode_mutate_lr']).detach()
            decoder_dna = util.perturb_type2(decoder_dna, kwargs['decode_mutate_prob']).detach()
        return DecoderGenotype(decoder_dna, [self], 'mutate')
    
class BreederGenotype():
    id_factory = 0
    def __init__(self, breeder_dna=None, genos_parent=None, orgin_type=None):
        self.breeder_dna = breeder_dna
        
        if genos_parent is not None:
            self.parents_id = [geno.id for geno in genos_parent]
        
        self.id = BreederGenotype.id_factory
        BreederGenotype.id_factory += 1
    
    def generate_random(**kwargs):
        bc, device = [kwargs[i] for i in ['breeder_class', 'device']]
        
        breeder_dna = util.model2vec(bc(**kwargs)).to(device)
        return BreederGenotype(breeder_dna, None, 'random')
        
    def clone(self):
        return BreederGenotype(self.breeder_dna, [self], 'clone')
    
    def mutate(self, **kwargs):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        breeder_dna = self.breeder_dna
        if breeder_dna is not None:
            breeder_dna = util.perturb_type1(breeder_dna, kwargs['breed_mutate_lr']).detach()
            breeder_dna = util.perturb_type2(breeder_dna, kwargs['breed_mutate_prob']).detach()
        return BreederGenotype(breeder_dna, [self], 'mutate')

class Neuroevolution:
    def __init__(self, geno_config, evol_config, device='cpu', verbose=False):
        self.geno_config = geno_config
        self.evol_config = evol_config
        self.device = device
        
        self.best_fitness = None
        self.fitness_v_gen_AUC = 0
        
        pwl = len(util.model2vec(geno_config['pheno_class']()).detach())
        geno_config['pheno_weight_len'] = pwl
        if geno_config['decoder_class'] is models_decode.IdentityDecoder:
            geno_config['dna_len'] = geno_config['pheno_weight_len']
            
        if verbose:
            print('Running Neuroevolution with ')
#             print('Population: ', evol_config['n_pop'])
            print('Generations: ', evol_config['n_gen'])
            pheno, decoder, breeder = self.init_pheno_decoder_breeder()
            n_params = len(util.model2vec(pheno))
            print(f'Pheno   # params: {n_params:05d}')
            print('------------------')
            n_params_total = 0
            n_params = geno_config['dna_len']; n_params_total += n_params
            print(f'DNA     # params: {n_params:05d}')
            n_params = len(util.model2vec(decoder)); n_params_total += n_params
            print(f'Decoder # params: {n_params:05d}')
            n_params = len(util.model2vec(breeder)); n_params_total += n_params
            print(f'Breeder # params: {n_params:05d}')
            print(f'Total   # params: {n_params_total:05d}')
        
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
        # TODO make this sample from breeders based on breeder fitness
        geno_breeders = np.random.choice(self.pop_breeder, size=len(pop1))
        return np.array([geno1.crossover(geno_breeder, geno2, breeder=breeder) 
                         for geno_breeder, geno1, geno2 in zip(geno_breeders, pop1, pop2)])
    
    def calc_fitdata_dna(self, pop_dna, pop_decoder, n_sample=10):
        fitdata = []
        pheno, decoder, _ = self.init_pheno_decoder_breeder()
        for geno in pop_dna:
            fitdata_i = []
            for geno_decoder in np.random.choice(pop_decoder, size=n_sample, replace=False):
                pheno = geno.to_pheno(geno_decoder, decoder, pheno)
                fitdata.append(self.evol_config['fitness_func'](pheno, device=self.device))
        fitdata = util.arr_dict_mean(np.array(fitdata).reshape((len(pop_dna), n_sample)), axis=-1) 
        for geno, fd in zip(pop_dna, fitdata):
            geno.fitdata = fd
        return util.arr_dict2dict_arr(fitdata)
        
    def calc_fitdata_decoder(self, pop_decoder, pop_dna, n_sample=10):
        fitdata = []
        pheno, decoder, _ = self.init_pheno_decoder_breeder()
        for geno_decoder in pop_decoder:
            fitdata_i = []
            for geno in np.random.choice(pop_dna, size=n_sample, replace=False):
                pheno = geno.to_pheno(geno_decoder, decoder, pheno)
                fitdata.append(self.evol_config['fitness_func'](pheno, device=self.device))
        fitdata = util.arr_dict_mean(np.array(fitdata).reshape((len(pop_decoder), n_sample)), axis=-1) 
        for geno_decoder, fd in zip(pop_decoder, fitdata):
            geno_decoder.fitdata = fd
        return fitdata
    
    def calc_fitdata_breeder(self, pop_breeder, pop_dna, pop_decoder, n_sample=20):
        fitdata = []
        pheno, decoder, breeder = self.init_pheno_decoder_breeder()
        for geno_breeder in pop_breeder:
            fitdata_i = []
            for geno1, geno2 in np.random.choice(pop_dna, size=(n_sample, 2), replace=True):
                geno_decoder = np.random.choice(pop_decoder)
                geno = geno1.crossover(geno_breeder, geno2, breeder)
                pheno = geno.to_pheno(geno_decoder, decoder, pheno)
                fd = self.evol_config['fitness_func'](pheno, device=self.device)
                fd = {key: fd[key]-(geno1.fitdata[key]+geno2.fitdata[key])/2. for key in fd.keys()}
                fitdata.append(fd)
        fitdata = util.arr_dict_mean(np.array(fitdata).reshape((len(pop_decoder), n_sample)), axis=-1) 
        for geno_breeder, fd in zip(pop_breeder, fitdata):
            geno_breeder.fitdata = fd
        return fitdata
    
    def run_evolution(self, tqdm=None, logger=None, tag=None):
        
        self.npop_dna = np.array([DNAGenotype.generate_random(device=self.device, **self.geno_config)
                                  for _ in range(self.evol_config['n_pop_dna'])])
        self.npop_decoder = np.array([DecoderGenotype.generate_random(device=self.device, **self.geno_config)
                                  for _ in range(self.evol_config['n_pop_decoder'])])
        self.npop_breeder = np.array([BreederGenotype.generate_random(device=self.device, **self.geno_config)
                                  for _ in range(self.evol_config['n_pop_breeder'])])
        
        
        loop = range(self.evol_config['n_gen'])
        for gen_idx in (tqdm(loop) if tqdm is not None else loop):
            self.pop_dna = self.npop_dna
            self.pop_decoder = self.npop_decoder
            self.pop_breeder = self.npop_breeder
            
            self.fitdata = self.calc_fitdata_dna(self.pop_dna, self.pop_decoder)
            self.calc_fitdata_decoder(self.pop_decoder, self.pop_dna)
            self.calc_fitdata_breeder(self.pop_breeder, self.pop_dna, self.pop_decoder)
            
            fit = np.array([g.fitdata['fitness'] for g in self.pop_dna])
            prob = population.fit2prob_sm(fit, **self.evol_config)
            self.npop_dna = population.calc_next_population(self.pop_dna, prob, 
                                                            self.calc_clone, self.calc_mutate, 
                                                            self.calc_crossover, **self.evol_config)
            fit = np.array([g.fitdata['fitness'] for g in self.pop_decoder])
            prob = population.fit2prob_sm(fit, **self.evol_config)
            self.npop_decoder = population.calc_next_population(self.pop_decoder, prob, 
                                                            self.calc_clone, self.calc_mutate, 
                                                            calc_crossover=None, **self.evol_config)
            fit = np.array([g.fitdata['fitness'] for g in self.pop_breeder])
            prob = population.fit2prob_sm(fit, **self.evol_config)
            self.npop_breeder = population.calc_next_population(self.pop_breeder, prob, 
                                                            self.calc_clone, self.calc_mutate, 
                                                            calc_crossover=None, **self.evol_config)
            
            if logger is not None:
                self.log_stats(gen_idx, logger, tag)

    def log_stats(self, gen_idx, logger, tag=None):
        logger.add_scalar(f'{tag}/gpu_mem_allocated', torch.cuda.memory_allocated(), global_step=gen_idx)
        
        self.best_fitness = np.max(self.fitdata['fitness']) if self.best_fitness is None else max(self.best_fitness, np.max(self.fitdata['fitness']))
                
        self.fitness_v_gen_AUC += np.max(self.fitdata['fitness'])
        
        fitdata = util.arr_dict2dict_arr(np.array([g.fitdata for g in self.pop_breeder]))
        for key, value in fitdata.items():
            data = np.array(value)
#             data = data[data>=np.median(data)]
            logger.add_histogram(f'{tag}/breeder_{key}', data, global_step=gen_idx)
    
        fitdata = util.arr_dict2dict_arr(np.array([g.fitdata for g in self.pop_decoder]))
        for key, value in fitdata.items():
            data = np.array(value)
#             data = data[data>=np.median(data)]
            logger.add_histogram(f'{tag}/decoder_{key}', data, global_step=gen_idx)
            
        for key, value in self.fitdata.items():
            data = np.array(value)
#             data = data[data>=np.median(data)]
            logger.add_histogram(f'{tag}/{key}', data, global_step=gen_idx)
            
#         logger.add_histogram(f'{tag}/prob_of_selection', self.prob, global_step=gen_idx)
        
#         if self.pop_dna[0].dna is not None:
#             all_dna_weights = torch.cat([geno.dna for geno in self.pop_dna]).detach().cpu()
#             logger.add_histogram(f'{tag}/all_dna_weights', all_dna_weights, global_step=gen_idx)
#         if self.pop_decoder[0].decoder_dna is not None:
#             all_decoder_dna_weights = torch.cat([geno.decoder_dna for geno in self.pop_decoder]).detach().cpu()
#             if all_decoder_dna_weights.numel()>0:
#                 logger.add_histogram(f'{tag}/all_decoder_dna_weights', all_decoder_dna_weights, global_step=gen_idx)
#         if self.pop_breeder[0].breeder_dna is not None:
#             all_breeder_dna_weights = torch.cat([geno.breeder_dna for geno in self.pop_breeder]).detach().cpu()
#             if all_breeder_dna_weights.numel()>0:
#                 logger.add_histogram(f'{tag}/all_breeder_dna_weights', all_breeder_dna_weights, global_step=gen_idx)
        
#         torch.save(self.pop, os.path.join(logger.log_dir, f'pop_gen_{gen_idx:05d}.pop'))
#         torch.save(self.fitdata, os.path.join(logger.log_dir, f'fitdata_gen_{gen_idx:05d}'))
        
        if gen_idx==0:
            torch.save(self.geno_config, os.path.join(logger.log_dir, 'geno_config'))
        
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
    
    
    