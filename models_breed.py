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
    
class NonlinearBreeder(Breeder):
    def __init__(self, **kwargs):
        super().__init__()
        d_in = kwargs['dna_len']

        self.lin1 = nn.Linear(n, n)
        self.lin2 = nn.Linear(2*n, n)
        self.lin3 = nn.Linear(n, n)
        
        if kwargs['breed_weight_init_for_first_parent']:
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
        x = torch.tanh(x)
        return x

    def breed_dna(self, dna1, dna2):
        return self(torch.stack([dna1, dna2], dim=0)[None])[0]
    
    def generate_random(config, device='cpu'):
        return nn.utils.parameters_to_vector(NonlinearBreeder(config).parameters()).detach().to(device)
    
    def load_breeder_dna(self, breeder_dna):
        nn.utils.vector_to_parameters(breeder_dna, self.parameters())