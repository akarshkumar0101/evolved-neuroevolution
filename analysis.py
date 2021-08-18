import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import util
from tqdm import tqdm
import random

def get_mrs_fitness(x, mrs, optim_fn, n_samples=None):
    for _ in range(x.ndim):
        mrs = mrs[..., None]
    shape = list(x.shape)
    if n_samples is not None:
        shape = [n_samples] + shape
        mrs = mrs[..., None]
    x_aft = x+torch.randn(shape).to(x)*mrs
    x_bef = x
    fit_bef = optim_fn(x_bef)
    fit_aft = optim_fn(x_aft)
    fit_diff = fit_aft-fit_bef
    return x_bef, x_aft, fit_diff

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


def draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=True, annotate=True):
    best_mrs = mrs[fit_diff.min(dim=-1).values.argsort()]
    a, b = torch.meshgrid(torch.from_numpy(fd_bins).float(), mrs)
    plt.pcolormesh(a, b, ns.T/ns.sum(axis=-1), shading='auto', 
                   norm=LogNorm() if log_dist else None)
    
    if annotate:
        plt.axvline(c='y', label='$\delta f$ = zero')
        # for i in range(1):
        #     plt.plot(fit_diff.sort(dim=-1).values[:, i], mrs, c='g', 
        #              label='best' if i==0 else None)
        for i, ii in enumerate([10, 50, 500, 1000, 5000]):
            plt.plot(fit_diff.sort(dim=-1).values[:, :ii].mean(dim=-1), mrs, c='c', 
                     label='lower avg' if i==0 else None)
#         plt.plot(fit_diff.reshape(-1, 100, 100).min(dim=-1).values.mean(dim=-1), mrs, 
#                  c='b', label='avg of mins')
        for i in range(10):
            plt.axhline(best_mrs[i], c='magenta', label='best MRs' if i==0 else None)
        plt.plot(fit_diff.min(dim=-1).values, mrs, c='orangered', label='min of $\delta f$')
        plt.plot(fit_diff.max(dim=-1).values, mrs, c='orangered', label='max of $\delta f$')
        plt.plot(fit_diff.mean(dim=-1), mrs, c='r', label='mean of $\delta f$')
        plt.legend()
    plt.colorbar()
    plt.yscale('log')
    plt.xlim(fd_bins.min(), fd_bins.max())
    
    plt.ylabel('Mutation Rate, $\mu$')
    plt.xlabel('Delta (after-before) Fitness of Mutation, $\delta(x, \mu)$')
    
def viz_mrs_performance(mrs, fd_bins, ns, fit_diff):
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    plt.sca(axs[0])
    draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=False, annotate=False)
    plt.title('MR Distribution over Delta Fitness')
    plt.sca(axs[1])
    draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=True, annotate=False)
    plt.title('Log Distribution')
    plt.sca(axs[2])
    draw_mrs_performance(mrs, fd_bins, ns, fit_diff, log_dist=True, annotate=True)
    return fig, axs
    
