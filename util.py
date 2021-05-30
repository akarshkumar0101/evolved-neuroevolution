import numpy as np
import torch
from torch import nn


def to_np_obj_array(a):
    """
    Takes in a list/iterable of stuff and turns it into a numpy object array.
    """
    na = np.empty(len(a), dtype=object)
    for i, item in enumerate(a):
        na[i] = item
    return na

def perturb_type1(a, lr):
    return a + lr*torch.randn_like(a)

def perturb_type2(a, prob, rgn=torch.randn):
    a = a.clone()
    mask = torch.rand_like(a)<prob
#     a[mask] = 1e-1*torch.randn_like(a)[mask]
    a[mask] = 1e-1*rgn(mask.sum()).to(a)
    return a


def model2vec(model, device='cpu'):
    p = model.parameters()
    n_params = np.sum([pi.numel() for pi in p], dtype=int)
    if n_params==0:
        return torch.tensor([])
    return nn.utils.parameters_to_vector(model.parameters())

def vec2model(v, model):
    p = model.parameters()
    n_params = np.sum([pi.numel() for pi in p], dtype=int)
    if len(v)!=n_params:
        raise Exception('Not correct number of parameters')
    nn.utils.vector_to_parameters(v, model.parameters())
    return model
