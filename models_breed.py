import numpy as np
import torch
import torch.nn as nn

class Breeder(nn.Module):
    def __init__(self, config=None):
        super().__init__()

class FirstParentIdentityBreeder(Breeder):
    def __init__(self, **kwargs):
        super().__init__()
    
    def breed_dna(self, dna1, dna2):
        return dna1
    
    def load_breeder_dna(self, breeder_dna):
        pass

class LinearBreeder(Breeder):
    def __init__(self, **kwargs):
        super().__init__()
        d_in = kwargs['dna_len']
        
        self.lin1 = nn.Linear(2*d_in, d_in)
        
        if kwargs['breed_weight_init_for_first_parent']:
            self.init_weights_for_first_parent()
        
    def init_weights_for_first_parent(self):
        for lin in [self.lin1]:
            lin.weight.data = torch.eye(*lin.weight.shape).to(lin.weight)
            lin.bias.data = torch.zeros_like(lin.bias)
        
    def forward(self, x):
        bs = len(x)
        x = self.lin1(x)
        return x

    def breed_dna(self, dna1, dna2):
        return self(torch.cat([dna1, dna2])[None])[0]
    
class NonlinearBreederSmall(Breeder):
    def __init__(self, **kwargs):
        super().__init__()
        d_in = kwargs['dna_len']

        ls = np.linspace(d_in*2, d_in, num=2, dtype=int)
        self.lins = nn.ModuleList([nn.Linear(i, j) for i, j in zip(ls[:-1], ls[1:])])
        
        if kwargs['breed_weight_init_for_first_parent']:
            self.init_weights_for_first_parent()
        
    def init_weights_for_first_parent(self):
        for lin in [self.lin1, self.lin2, self.lin3]:
            lin.weight.data = torch.eye(*lin.weight.shape).to(lin.weight)
            lin.bias.data = torch.zeros_like(lin.bias)

    def forward(self, x):
        bs = len(x)
        x = x.reshape(bs, -1)
        for lin in self.lins:
            x = torch.tanh(lin(x))
        return x

    def breed_dna(self, dna1, dna2):
        return self(torch.stack([dna1, dna2], dim=0)[None])[0]
    