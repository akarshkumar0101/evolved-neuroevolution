import numpy as np
import matplotlib.pyplot as plt

import util
from tqdm import tqdm

def plot_generation_fitnesses(y, x=None, show_dist=True, c=None, label=None):
    """
    `y` should be of shape (time points, num samples)
    """
    if x is None:
        x = np.arange(len(y))
    y_avg, y_std = y.mean(axis=-1), y.std(axis=-1)
    y_max, y_min = y.max(axis=-1), y.min(axis=-1)
    
    if show_dist:
        plt.errorbar(x=x, y=(y_max+y_min)/2., yerr=(y_max-y_min)/2., elinewidth=0.5, c=c)
        plt.plot(x, y_max, c=c)
    plt.plot(x, y_min, c=c, label=label)
    
def to_pheno_mix_breed_decode(ne, decoder_geno, dna1_geno, dna2_geno=None, breeder_geno=None):
    if breeder_geno is None:
        gneo = dna1_geno
    else:
        geno_dna = dna1_geno.crossover(dna2_geno, breeder_geno, ne.breeder)
    return geno_dna.load_pheno(decoder_geno, ne.decoder, ne.pheno)

def to_pheno_mix_breed_decode_combined(ne, decoder_cgeno, dna1_cgeno, dna2_cgeno=None, breeder_cgeno=None):
    a, b= decoder_cgeno.geno_decoder, dna1_cgeno.geno_dna
    c = dna2_cgeno.geno_dna if dna2_cgeno is not None else None
    d = breeder_cgeno.geno_breeder if breeder_cgeno is not None else None
    return to_pheno_mix_breed_decode(ne, a, b, c, d)

def eval_evolution_with_good_geno(ne, decode_kw, dna1_kw, dna2_kw, breeder_kw, n_samp=10):
    """
    This method lets you use already evolved DNA/decoder/breeder on evolving DNA/decoder/breeder
    to evaluate how their fitness changes throughout generations.
    decode_kw, dna1_kw, dna2_kw, breeder_kw can be one of
     - 'evol1': use randomly sampled evolving version
     - 'evol2': use randomly sampled evolving version #2
     - 'good1': use randomly sampled evolved version
     - 'good2': use randomly sampled evolved version #2
     - 'none': ignore
    """
    fitdata = []
    gen_idx_good = max(ne.pop_evol.keys())
    pop_good = ne.pop_evol[gen_idx_good]
    for gen_idx in tqdm(ne.pop_evol.keys(), leave=False):
        pop_evol = ne.pop_evol[gen_idx]
        for i in range(n_samp):
            d = {'none': None}
            d['evol1'], d['evol2'] = np.random.choice(pop_evol, size=(2,), )
            d['good1'], d['good2'] = np.random.choice(pop_good, size=(2,), )
            pheno = to_pheno_mix_breed_decode_combined(ne, d[decode_kw], d[dna1_kw], d[dna2_kw], d[breeder_kw])
            fd = ne.evol_cfg['fitness_func'](pheno, device=ne.device)
            fitdata.append(fd)
    return fitdata

def viz_evolution_with_good_geno(gens, fitdata):
    plt.figure(figsize=(20, 10))
    x = np.tile(gens[:, None], (1, fitdata.shape[-1]));y = fitdata
    plt.subplot(211)
    plt.scatter(x.flatten(), y.flatten(), s=2, c='r')
    plt.errorbar(x=x.mean(axis=-1), y=y.mean(axis=-1), yerr=y.std(axis=-1), elinewidth=1)
    plt.subplot(212)
    plt.scatter(x.flatten(), y.flatten(), s=2, c='r')
    plt.errorbar(x=x.mean(axis=-1), y=y.mean(axis=-1), yerr=y.std(axis=-1), elinewidth=1)
    plt.ylim(-2.3, -.8)
    