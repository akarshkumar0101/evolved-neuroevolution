import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

import util
from tqdm import tqdm
import random

def plot_mean_std(f, name, c, logscale=False, 
                  render_mean=True, render_std=True, render_plots=True):
    if logscale:
        y = f.log().mean(dim=0).exp()
        yerr = f.log().std(dim=0).exp()
    else:
        y = f.mean(dim=0)
        yerr = f.std(dim=0)
        
#     plt.figure(figsize=(10, 5))
    x = np.arange(len(y))
    c = colors.hex2color(colors.cnames[c])
    if render_mean:
        plt.plot(x, y, color=c, label=name)
    c = list(c)+[0.2]
    if render_std:
        if logscale:
            plt.fill_between(x, y/yerr, y*yerr, color=c, label=name if not render_mean else None)
        else:
            plt.fill_between(x, y-yerr, y+yerr, color=c, label=name if not render_mean else None)
    if render_plots:
        for fi in f:
            plt.plot(x, fi, c=c)

    if logscale:
        plt.yscale('log')
    plt.legend()

class MultiPlot():
    def __init__(self):
        self.data = {}
    
    def plot(self, f, name, c, logscale=False):
        if name not in self.data.keys():
            self.data[name] = {'data': [], 'color': None, logscale:False}
        self.data[name]['data'].append(f)
        self.data[name]['color'] = c
        self.data[name]['logscale'] = logscale

    def render(self):
        for label, data in self.data.items():
            plot_mean_std(torch.stack(data['data']), label, data['color'], data['logscale'])
            
        
def plot_fits_vs_gens(y, x=None, show_error=True, c=None, label=None):
    """
    `y` should be of shape (time points, num samples)
    """
    if x is None:
        x = np.arange(len(y))
        
    if c is None:
        c = [random.random() for _ in range(3)]
        
    y_avg, y_std = y.mean(axis=-1), y.std(axis=-1)
    y_max, y_min = y.max(axis=-1), y.min(axis=-1)
    
    if show_error:
        plt.errorbar(x=x, y=(y_max+y_min)/2., yerr=(y_max-y_min)/2., elinewidth=0.5, c=c)
    plt.plot(x, y_max, c=c)
    plt.plot(x, y_min, c=c, label=label)
    plt.xlim(np.min(x), np.max(x))
    
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
    
    