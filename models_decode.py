import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
    def load_decoder_dna(self, decoder_dna):
        nn.utils.vector_to_parameters(decoder_dna, self.parameters())
    
    def decode_dna(self, dna):
        return self(dna[None])[0]
    
    def generate_random(config, device='cpu'):
        return nn.utils.parameters_to_vector(LinearDecoder(config).parameters()).detach().to(device)
    
    def load_decoder_dna(self, decoder_dna):
        nn.utils.vector_to_parameters(decoder_dna, self.parameters())

class IdentityDecoder(Decoder):
    def __init__(self, config=None):
        super().__init__()
        
    def forward(self, x):
        return x
    
    def generate_random(config, device='cpu'):
        return None
    
    def load_decoder_dna(self, decoder_dna):
        pass
        
    
class LinearDecoder(Decoder):
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

