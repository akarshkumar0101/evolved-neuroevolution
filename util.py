import numpy as np
import torch


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
