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