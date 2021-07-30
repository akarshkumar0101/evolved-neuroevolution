import numpy as np
import torch
import matplotlib.pyplot as plt

import util
from tqdm import tqdm
import random


def viz_mrs_fit_hists(mrs, x, xmut, fit_diff):
    plt.figure(figsize=(20, 10))
    for i, mr in enumerate(mrs):
        fd = fit_diff[i]
        low, high = fit_diff.min(), fit_diff.max()
        low, high = -10, 20
        bins = np.linspace(low, high, 100)
        plt.subplot(4, 4, i+1)
        plt.hist(fd.cpu().numpy(), bins=bins);
        plt.axvline(c='y', label='zero')
        plt.axvline(fd.mean(), c='r', label='mean')
    #     plt.axvline(fd.min(), c='g', label='min')
        for i in range(3):
            plt.axvline(np.sort(fd)[i], c='orangered', label='min' if i==0 else None)
        plt.axvline(fd.max(), c='orangered', label='max')
        for i, ii in enumerate([10, 50, 500, 1000, 5000]):
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
    return plt.gcf()
def calc_fd_bins_ns(mrs, fit_diff, fd_bins=None):
    if fd_bins is None:
        fd_bins = np.linspace(fit_diff.min(), fit_diff.max(), 101)
    ns = []
    for i, mr in enumerate(mrs):
        fd = fit_diff[i]
    #     n, _, chart = plt.hist(fd.cpu().numpy(), bins=bins)
        n, _  = np.histogram(fd.cpu().numpy(), bins=fd_bins)
        ns.append(n)
    ns = np.stack(ns)
    return ns
from matplotlib.colors import LogNorm
def viz_mrs_performance(mrs, fd_bins, ns, fit_diff):
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    best_mrs = mrs[fit_diff.min(dim=-1).values.argsort()]
    a, b = torch.meshgrid(torch.from_numpy(fd_bins).float(), mrs)
    plt.subplot(131)
    plt.pcolormesh(a, b, ns.T/ns.sum(axis=-1), shading='auto')
    plt.colorbar()
    plt.yscale('log')
    plt.title('MR Distribution over Delta Fitness')
    plt.subplot(132)
    plt.pcolormesh(a, b, ns.T/ns.sum(axis=-1), shading='auto', norm=LogNorm())
    plt.colorbar()
    plt.yscale('log')
    plt.title('Log Distribution')
    plt.subplot(133)
    plt.pcolormesh(a, b, ns.T/ns.sum(axis=-1), shading='auto', norm=LogNorm())
    plt.axvline(c='y', label='zero')
    # for i in range(1):
    #     plt.plot(fit_diff.sort(dim=-1).values[:, i], mrs, c='g', 
    #              label='best' if i==0 else None)
    for i, ii in enumerate([10, 50, 500, 1000, 5000]):
        plt.plot(fit_diff.sort(dim=-1).values[:, :ii].mean(dim=-1), mrs, c='c', 
                 label='lower avg' if i==0 else None)
    plt.plot(fit_diff.reshape(-1, 100, 100).min(dim=-1).values.mean(dim=-1), mrs, c='b', label='avg of mins')
    for i in range(10):
        plt.axhline(best_mrs[i], c='magenta', label='best MRs' if i==0 else None)
    plt.plot(fit_diff.min(dim=-1).values, mrs, c='orangered', label='min')
    plt.plot(fit_diff.max(dim=-1).values, mrs, c='orangered', label='max')
    plt.plot(fit_diff.mean(dim=-1), mrs, c='r', label='mean')
    plt.colorbar(); plt.legend()
    plt.yscale('log')
    plt.title('Annotated')
    plt.xlim(fd_bins.min(), fd_bins.max())
    
    for ax in axs:
        ax.set_ylabel('Mutation Rate')
        ax.set_xlabel('Delta Fitness (after-before)')
    return fig, axs
    