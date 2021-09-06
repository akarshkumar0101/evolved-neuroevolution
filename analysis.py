import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import util
from tqdm import tqdm
import random

from scipy.signal import argrelextrema

def get_mrs_fitness(x, mrs, optim_fn, n_samples=None):
    for _ in range(x.ndim):
        mrs = mrs[..., None]
    shape = list(x.shape)
    if n_samples is not None:
        shape = [n_samples] + shape
        mrs = mrs[..., None]
    shape = [len(mrs)]+shape # don't share noise over MRs
    xmut = x+torch.randn(shape).to(x)*mrs
    fit_diff = optim_fn(xmut) - optim_fn(x)
    return x, xmut, fit_diff

def get_optimal_mr_extreme(x, mrs, optim_fn, n_samples=None):
    x, xmut, fit_diff = get_mrs_fitness(x, mrs, optim_fn, n_samples)
    return mrs[fit_diff.min(dim=-1).values.argmin(dim=0)]

def get_optimal_mr_look_ahead(x, mrs, optim_fn, n_gen=10, n_samples=1):
    if x.ndim==2:
        x = x[None]
    f = []
    for xi in x:
        for mr in mrs:
            for i in range(n_samples):
                pops, fits = optim.run_evolution_base(xi, optim_fn, n_gen, mr=mr)
                f.append(fits.min())
    f = torch.stack(f).reshape(len(x), len(mrs), n_samples)
    return mrs[f.mean(dim=-1).argmin(dim=-1)]

def viz_mrs_fit_hists(mrs, x, xmut, fit_diff, axshape=None, 
                      draw_n_min=3, draw_mins_avg=[10, 50, 500, 1000]):
    if axshape is None:
        axshape = (len(mrs), 1)
    
    fig, axs = plt.subplots(*axshape, figsize=(20, 10))
    for i, (ax, mr) in enumerate(zip(axs.flatten(), mrs)):
        plt.sca(ax)
        fd = fit_diff[i]
        low, high = fit_diff.min(), fit_diff.max()
        low, high = -10, 20
        bins = np.linspace(low, high, 100)
        plt.hist(fd.cpu().numpy(), bins=bins);
        plt.axvline(c='y', label='zero')
        plt.axvline(fd.mean(), c='r', label='mean')
    #     plt.axvline(fd.min(), c='g', label='min')
        for i in range(draw_n_min):
            plt.axvline(np.sort(fd)[i], c='orangered', label='min' if i==0 else None)
        plt.axvline(fd.max(), c='orangered', label='max')
        for i, ii in enumerate(draw_mins_avg):
            plt.axvline(fd.sort(dim=-1).values[:ii].mean(dim=-1), c='c', 
                     label='mins avg' if i==0 else None)
        plt.title(f'MR: {mr:.01e}, best: {fd.min():.01e}')
        plt.ylabel('# of mutations')
        plt.xlabel('Delta Fitness (after-before)')
        plt.xlim(low, high)
        plt.ylim(0,3000)
        plt.legend()
    plt.suptitle('PDF of Gain in Fitness for a mutation')
    plt.tight_layout()
    return fig, axs


def draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=True, annotate=True, top_legend=False, cbar=True):
    best_mrs = mrs[fit_diff.min(dim=-1).values.mean(dim=-1).argsort()]
    best_mrs_mean = mrs[fit_diff.mean(dim=-1).mean(dim=-1).argsort()]
    a, b = torch.meshgrid(torch.from_numpy(fd_bins).float(), mrs)
    plt.pcolormesh(a, b, ns.T/ns.sum(axis=-1), shading='auto', 
                   norm=LogNorm() if log_dist else None)
    
    if annotate:
        plt.axvline(c='y', label='$\Delta=0$', linewidth=5)
        # for i in range(1):
        #     plt.plot(fit_diff.sort(dim=-1).values[:, i], mrs, c='g', 
        #              label='best' if i==0 else None)
        
#         argmins = argrelextrema(fit_diff.min(dim=-1).values.numpy(), np.less)[0]
#         if len(argmins)>0:
#             mins = fit_diff.min(dim=-1).values[argmins]
#             argmins = argmins[np.argsort(mins)]
#             if type(argmins) is not np.ndarray:
#                 argmins = [argmins]
#             print(argmins[:5])
#             for i in argmins[:5]:
#                 plt.axhline(mrs[i], c='magenta', label='best $\sigma$s' if i==0 else None)
#         for i, ii in enumerate([10, 50, 500, 1000, 5000]):
#             plt.plot(fit_diff.sort(dim=-1).values[:, :ii].mean(dim=-1), mrs, c='c', 
#                      label='lower avg' if i==0 else None)

#         plt.plot(fit_diff.reshape(-1, 100, 100).min(dim=-1).values.mean(dim=-1), mrs, 
#                  c='b', label='avg of mins')
        plt.plot(fit_diff.min(dim=-1).values.mean(dim=-1), mrs, c='r', label='min of $\Delta$s', linewidth=5)
        plt.plot(fit_diff.max(dim=-1).values.mean(dim=-1), mrs, c='r', label='max of $\Delta$s', linewidth=5)
        plt.plot(fit_diff.mean(dim=-1).mean(dim=-1), mrs, c='dodgerblue', label='mean of $\Delta$s', linewidth=5)
#         plt.plot(fit_diff.mean(dim=-1)-fit_diff.var(dim=-1), mrs, c='k', label='mean-var')
#         plt.plot(fit_diff.mean(dim=-1)-fit_diff.std(dim=-1), mrs, c='gray', label='mean-std')
        for i in range(1):
            plt.axhline(best_mrs[i], c='purple', label='$\sigma=\sigma^*_{min}$' if i==0 else None, linewidth=5)
        for i in range(1):
            plt.axhline(best_mrs_mean[i+2], c='olive', label='$\sigma=\sigma^*_{\mu}$' if i==0 else None, linewidth=5)
        if top_legend:
            plt.legend(fontsize=20, bbox_to_anchor=(0, 1.9), loc='upper left', ncol=3)
        else:
            plt.legend(fontsize=30, bbox_to_anchor=(1.37, 1.2), loc='upper left')
        handles, labels = plt.gca().get_legend_handles_labels()
        
    plt.locator_params(axis='y', nbins=2); plt.locator_params(axis='x', nbins=2)
    plt.xticks(fontsize=20); plt.yticks(fontsize=20)
        
    if cbar:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
    plt.yscale('log')
    plt.xlim(fd_bins.min(), fd_bins.max())
    
    if annotate:
        return handles, labels
    
    
def viz_mrs_performance(mrs, fd_bins, ns, fit_diff, b=False):
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    plt.sca(axs[0])
    draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=False, annotate=False)
    plt.title('Distribution', fontsize=30)
    plt.ylabel('Mutation Rate, $\sigma$', fontsize=35)
    plt.sca(axs[1])
    draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=True, annotate=False)
    plt.title('Log Distribution', fontsize=30)
    plt.xlabel('Function Value Change of Mutation, $\Delta(x, \sigma)$', fontsize=35)
    plt.sca(axs[2])
    draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=True, annotate=True)
    plt.title('Annotated Log Distribution', fontsize=30)
    if b:
        plt.tight_layout()
    return fig, axs
    
