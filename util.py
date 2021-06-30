import numpy as np
import torch
from torch import nn

from functools import partial

def to_np_obj_array(a):
    """
    Takes in a list/iterable of stuff and turns it into a numpy object array.
    """
    na = np.empty(len(a), dtype=object)
    for i, item in enumerate(a):
        na[i] = item
    return na

def uniform_crossover(a, b, p=.5):
    c = a.clone()
    mask = torch.rand_like(a)<p
    c[mask] = b[mask]
    return c

def partial_additive_corrupting_noise(a, p, eps, add=True, rgn=torch.randn_like):
    a = a.clone()
    mask = torch.rand_like(a)<p
    noise = eps*rgn(a[mask])
    if add:
        a[mask] = a[mask] + noise
    else:
        a[mask] = noise
    return a

partial_corrupting_noise = partial(partial_additive_corrupting_noise, add=False)
partial_additive_noise = partial(partial_additive_corrupting_noise, add=True)
additive_noise = partial(partial_additive_corrupting_noise, p=1, add=True)

def mask_inverse_noise(a, p, rgn=torch.randn):
    a = a.clone()
    mask = torch.rand_like(a)<p
    a[mask] = ~a[mask]
    return a
    
def count_params(model):
    return np.sum([p.numel() for p in model.parameters()], dtype=int)
    
def model2vec(model, use_torch_impl=True):
    if use_torch_impl:
        p = model.parameters()
        n_params = np.sum([pi.numel() for pi in p], dtype=int)
        if n_params==0:
            return torch.tensor([])
        return nn.utils.parameters_to_vector(model.parameters())
    else:
        return torch.cat([p.flatten() for p in model.parameters()])

def vec2model(v, model, use_torch_impl=True):
    n_params = count_params(model)
    if len(v)!=n_params:
        raise Exception('Not correct number of parameters')
        
    if use_torch_impl:
        nn.utils.vector_to_parameters(v, model.parameters())
    else:
        i = 0
        for p in model.parameters():
            l = p.numel()
            p.data = v[i:i+l].reshape(p.shape)
            i = i+l
    return model

def dict_list2list_dict(DL):
    return [dict(zip(DL,t)) for t in zip(*DL.values())]

def list_dict2dict_list(LD):
    return {k: np.array([dic[k] for dic in LD]) for k in LD[0]}

def dict_arr2arr_dict(DA):
    key = list(DA.keys())[0]
    res = np.empty(DA[key].size, dtype=object)
    for i in range(res.size):
        res[i] = {key: DA[key].ravel()[i] for key in DA.keys()}
    res = res.reshape(DA[key].shape)
    return res

def arr_dict2dict_arr(AD):
    res = {}
    e = AD.ravel()[0]
    for key in e.keys():
        res[key] = np.empty(AD.size, dtype=type(list(e.values())[0]))
        for i, el in enumerate(AD.ravel()):
            res[key][i] = el[key]
        res[key] = res[key].reshape(AD.shape)
    return res

def arr_dict_mean(AD, axis=-1):
    DA = arr_dict2dict_arr(AD)
    return dict_arr2arr_dict({key:DA[key].mean(axis=axis) for key in DA.keys()})

def dict_arr_mean(DA, axis=-1):
    return {key:DA[key].mean(axis=axis) for key in DA.keys()}

def calc_pairwise_cossim(d1, d2=None):
    if d2 is None:
        d2 = d1
    d1, d2 = d1[None], d2[:, None]
    return torch.cosine_similarity(d1, d2, dim=-1)
    
def calc_pairwise_corr(d1, d2=None):
    if d2 is None:
        d2 = d1
    d1, d2 = d1[None], d2[:, None]
    d1, d2 = d1-d1.mean(dim=-1, keepdim=True), d2-d2.mean(dim=-1, keepdim=True)
    return (d1*d2).sum(dim=-1)/torch.sqrt(d1.pow(2.).sum(dim=-1)*d2.pow(2.).sum(dim=-1))
    