import numpy as np
import torch
import torch.nn as nn

import util

class AGMutator(nn.Module):
    """ Adaptive eps Additive Gaussian Mutator """
    def __init__(self, eps, shape='one', adaptive_eps=False, **kwargs):
        super().__init__()
        self.ori_eps = eps
        if shape=='one':
            self.eps = torch.tensor(eps)
        else:
            self.eps = torch.full(size=shape, fill_value=eps)

        if adaptive_eps:
            self.eps = nn.Parameter(self.eps).requires_grad_(False)
        self.delta = kwargs['delta']

    def mutate(self, a):
        # eps = (self.eps-self.ori_eps).abs()+self.ori_eps
        # eps = torch.clamp(self.eps, .7e-2, 1.3e-2)
        eps = torch.clamp(self.eps, self.ori_eps-self.delta, self.ori_eps+self.delta)
        self.eps.data = eps

        # eps = torch.clamp(self.eps, min=self.ori_eps, max=None)
        return util.additive_noise(a, eps=eps)

class ConvMutator(nn.Module):
    def __init__(self, init_noise=True):
        super().__init__()

        self.seq = nn.Sequential(*[
            nn.Conv1d(2, 2, 3, padding=1),
            nn.Sigmoid(),
            nn.Conv1d(2, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Conv1d(1, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Conv1d(1, 1, 3, padding=1),
            nn.Tanh(),
        ])
        self.eps = 5e-2
        if init_noise:
            for mod in self.modules():
                if type(mod) is nn.Conv1d:
                    mod.weight.data = 5e-1*torch.randn_like(mod.weight)
                    mod.bias.data = torch.zeros_like(mod.bias)

    def mutate(self, a):
        x = torch.stack([a,torch.randn_like(a)], dim=0)
        x = self.seq(x[None])[0, 0]
        # return a + self.eps*x
        return self.eps*x
        # return util.additive_noise(a, eps=self.eps)



