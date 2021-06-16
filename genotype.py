import torch

import util

class Genotype():
    id_factory = {}
    def __init__(self, genos_parent=None, origin_type=None, empty=False):
        if not empty:
            if self.__class__ not in Genotype.id_factory:
                Genotype.id_factory[self.__class__] = 0
            self.id = Genotype.id_factory[self.__class__]
            Genotype.id_factory[self.__class__] += 1

            if genos_parent is not None:
                self.parents_id = [geno.id for geno in genos_parent]
            else:
                self.parents_id =[]
            self.origin_type = origin_type
    
    def empty_copy(self):
        geno = Genotype(empty=True)
        geno.id = self.id
        geno.parents_id = self.parents_id
        geno.origin_type = self.origin_type
        return geno
        
class GenotypeContinuousMaskedDNA(Genotype):
    def __init__(self, weights=None, mask=None, genos_parent=None, origin_type=None):
        super().__init__(genos_parent, origin_type)
        self.weights = weights.clip(-2., 2.)
        self.mask = mask
    
    def generate_random(**kwargs):
        pc, device = [kwargs[i] for i in ['pheno_class', 'device']]
        
        dna = util.model2vec(pc(**kwargs)).to(device)
        mask = torch.zeros_like(dna, dtype=bool)
        return GenotypeContinuousDNA(dna, None, 'random')
        
    def clone(self):
        return GenotypeContinuousDNA(self.weights, self.mask, [self], 'clone')
    
    def mutate(self, **kwargs):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        weights = util.perturb_type1(self.weights, kwargs['dna_mutate_lr']).detach()
        mask = util.perturb_type3(self.mask, kwargs['dna_mutate_prob']).detach()
        
        return GenotypeContinuousDNA(weights, mask, [self], 'mutate')
    
    def crossover(self, geno2, geno_breeder, breeder):
        weights = self.weights
        mask = self.mask
        
        m = torch.rand_like(weights)<.5
        weights[m] = geno2.weights[m]
        m = torch.rand_like(mask)<.5
        mask[m] = geno2.mask[m]
        return GenotypeContinuousDNA(weights, mask, [self, geno2], 'breed')
    
    def load_pheno(self, pheno):
        w = torch.zeros_like(self.weights)
        w[self.mask] = self.weights[self.mask]
        pheno = util.vec2model(w, pheno)
        return pheno

class GenotypeContinuousDNA(Genotype):
    def __init__(self, dna=None, genos_parent=None, origin_type=None):
        super().__init__(genos_parent, origin_type)
        self.dna = dna.clip(-2., 2.)
    
    def generate_random(**kwargs):
        pc, device = [kwargs[i] for i in ['pheno_class', 'device']]
        
        # TODO this is not proper initialization of pheno_dna/dna
        if kwargs['dna_len']==kwargs['pheno_weight_len']:
            dna = util.model2vec(pc(**kwargs)).to(device)
        else:
            dna = kwargs['dna_init_std']*torch.randn(kwargs['dna_len']).to(device)
        return GenotypeContinuousDNA(dna, None, 'random')
        
    def clone(self):
        return GenotypeContinuousDNA(self.dna, [self], 'clone')
    
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
        return GenotypeContinuousDNA(dna, [self], 'mutate')
    
    def crossover(self, geno2, geno_breeder, breeder):
        breeder = geno_breeder.load_breeder(breeder)
        dna = breeder.breed_dna(self.dna, geno2.dna)
        return GenotypeContinuousDNA(dna, [self, geno2], 'breed')
    
    def load_pheno(self, geno_decoder, decoder, pheno):
        decoder = geno_decoder.load_decoder(decoder)
        pheno_weights = decoder.decode_dna(self.dna)
        pheno = util.vec2model(pheno_weights, pheno)
        return pheno
    
class GenotypeDecoderWeights(Genotype):
    def __init__(self, weights=None, genos_parent=None, origin_type=None):
        super().__init__(genos_parent, origin_type)
        self.weights = weights
        
    def generate_random(**kwargs):
        dc, device = [kwargs[i] for i in ['decoder_class', 'device']]
        
        weights = util.model2vec(dc(**kwargs)).to(device)
        return GenotypeDecoderWeights(weights, None, 'random')
        
    def clone(self):
        return GenotypeDecoderWeights(self.weights, [self], 'clone')
    
    def mutate(self, **kwargs):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        weights = self.weights
        if weights is not None:
            weights = util.perturb_type1(weights, kwargs['decode_mutate_lr']).detach()
            weights = util.perturb_type2(weights, kwargs['decode_mutate_prob']).detach()
        return GenotypeDecoderWeights(weights, [self], 'mutate')
    
    def load_decoder(self, decoder):
        return util.vec2model(self.weights, decoder)
    
    
class GenotypeBreederWeights(Genotype):
    def __init__(self, weights=None, genos_parent=None, origin_type=None):
        super().__init__(genos_parent, origin_type)
        self.weights = weights
    
    def generate_random(**kwargs):
        bc, device = [kwargs[i] for i in ['breeder_class', 'device']]
        weights = util.model2vec(bc(**kwargs)).to(device)
        return GenotypeBreederWeights(weights, None, 'random')
        
    def clone(self):
        return GenotypeBreederWeights(self.weights, [self], 'clone')
    
    def mutate(self, **kwargs):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        weights = self.weights
        if weights is not None:
            weights = util.perturb_type1(weights, kwargs['breed_mutate_lr']).detach()
            weights = util.perturb_type2(weights, kwargs['breed_mutate_prob']).detach()
        return GenotypeBreederWeights(weights, [self], 'mutate')
    
    def load_breeder(self, breeder):
        return util.vec2model(self.weights, breeder)
    
class GenotypeCombined(Genotype):
    def __init__(self, geno_dna, geno_decoder=None, geno_breeder=None, 
                 genos_parent=None, origin_type=None):
        super().__init__(genos_parent, origin_type)
        
        self.geno_dna = geno_dna
        self.geno_decoder = geno_decoder
        self.geno_breeder = geno_breeder
    
    def generate_random(**kwargs):
        kwargs['geno_class'] = GenotypeContinuousDNA
        kwargs['geno_decoder_class'] = GenotypeDecoderWeights
        kwargs['geno_breeder_class'] = GenotypeBreederWeights
        gc, gdc, gbc = [kwargs[i] for i in 
                        ['geno_class', 'geno_decoder_class', 'geno_breeder_class']]
        g = gc.generate_random(**kwargs)
        gd = gdc.generate_random(**kwargs)
        gb = gbc.generate_random(**kwargs)
        return GenotypeCombined(g, gd, gb, None, 'random')
        
    def clone(self):
        return GenotypeCombined(self.geno_dna.clone(), self.geno_decoder.clone(), self.geno_breeder.clone(), [self], 'clone')
    
    def mutate(self, **kwargs):
        """
        Mutation consists of 
         - a small perturbation to the solution
         - some extreme resets of some dimensions
        """
        g = self.geno_dna.mutate(**kwargs)
        gd = self.geno_decoder.mutate(**kwargs)
        gb = self.geno_breeder.mutate(**kwargs)
        return GenotypeCombined(g, gd, gb, [self], 'mutate')
    
    def crossover(self, geno2, breeder):
        geno_dna = self.geno_dna.crossover(geno2.geno_dna, self.geno_breeder, breeder)
        return GenotypeCombined(geno_dna, self.geno_decoder, self.geno_breeder, [self, geno2], 'breed')
        
    def load_pheno(self, decoder, pheno):
        return self.geno_dna.load_pheno(self.geno_decoder, decoder, pheno)
    