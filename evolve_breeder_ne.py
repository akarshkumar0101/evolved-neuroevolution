import models_pheno, models_breed
import torch
import xarray as xr
import numpy as np
import argparse

from tqdm import tqdm

import util
import mnist

import sys

parser = argparse.ArgumentParser(description='Run basic neuroevolution.')
parser.add_argument('--n-pop', type=int, default=100)
parser.add_argument('--n-gen', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu')

def run_evolution(args):
    task = mnist.MNIST()
    task.load_all_data(args.device)

    net = models_pheno.BigConvNet().to(args.device)
#     conet = models_breed.ConvRSBProbBreeder().to(args.device)
    conet = models_breed.ConvAdvancedRSBBreeder().to(args.device)
    triu_mask = torch.triu(torch.ones([args.n_pop]*2), diagonal=1).to(bool)
    pop = [(torch.randn(util.count_params(net)).to(args.device)/6.,
            torch.randn(util.count_params(conet)).to(args.device)/6.) for _ in range(args.n_pop)]
    fitdata = []
    loop = tqdm(range(args.n_gen), leave=False)
    for i in loop:
#         a = torch.stack(pop)
#         cossim = torch.cosine_similarity(a[None], a[:, None], dim=-1)[triu_mask].cpu().numpy()
#         l2 = (a[None]-a[:, None]).norm(dim=-1).mean()
#         l2sq = (a[None]-a[:, None]).norm(dim=-1).pow(2.).mean()

        fitdata.append(task.calc_pop_fitness(pop, lambda x: util.vec2model(x[0], net), 
                                             n_samples=1000, device=args.device))

        idx_selected = np.argsort(fitdata[-1].sel(metric='fitness'))[::-1][:30].data
        pop_selected = [pop[i] for i in idx_selected]
        npop = [pop_selected[0]]
        for _ in range(args.n_pop-1):
            a, b = np.random.choice(len(pop_selected), size=(2), replace=False)
            a, b = pop_selected[a], pop_selected[b]
            conet = util.vec2model(a[1], conet)
            c = conet.breed(a[0], b[0])
            c = util.additive_noise(c, eps=1e-2)
            npop.append((c.detach(), a[1]))
        pop = npop
        
        loop.set_postfix({'loss': fitdata[-1].sel(metric='loss').min().item()})
    fitdata = xr.concat(fitdata, dim='gen')
    return fitdata
    
args = parser.parse_args()

print(args)
fitdata = run_evolution(args)
torch.save(fitdata, f'data/runs/{args.seed}.ans')



