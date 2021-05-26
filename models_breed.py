import torch
import torch.nn as nn

class FirstParentIdentityBreeder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
    def breed_dna(self, dna1, dna2):
        return dna1

class LinearBreeder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config['dna_len']
        
        self.lin1 = nn.Linear(2*d_in, d_in)
        
        if config['breed_weight_init_for_first_parent']:
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
    
    
class NonlinearBreeder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config['dna_len']

        self.lin1 = nn.Linear(n, n)
        self.lin2 = nn.Linear(2*n, n)
        self.lin3 = nn.Linear(n, n)
        
        if config['breed_weight_init_for_first_parent']:
            self.init_weights_for_first_parent()
        
    def init_weights_for_first_parent(self):
        for lin in [self.lin1, self.lin2, self.lin3]:
            lin.weight.data = torch.eye(*lin.weight.shape).to(lin.weight)
            lin.bias.data = torch.zeros_like(lin.bias)
        

    def forward(self, x):
        bs = len(x)
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = x.reshape(bs, -1)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
#         return (a+b)/2. + x*self.amount
        return x

    def breed_dna(self, dna1, dna2):
        return self(torch.stack([dna1, dna2], dim=0)[None])[0]