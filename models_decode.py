import torch
import torch.nn as nn

class IdentityDecoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
    def decode_dna(self, dna):
        return dna

class LinearDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config['dna_len']
        d_out = config['pheno_weight_len']

        self.lin1 = nn.Linear(d_in, d_out)
        
        if config['decode_weight_init_zeros']:
            self.init_weights_for_zeros()
        
    def init_weights_for_zeros(self):
        for lin in [self.lin1]:
            lin.weight.data = torch.zeros_like(lin.weight)
            lin.bias.data = torch.zeros_like(lin.bias)
        
    def forward(self, x):
        bs = len(x)
        x = self.lin1(x)
        return x

    def decode_dna(self, dna):
        return self(dna[None])[0]