import torch
import numpy as np

def ackley(x):
    d = x.shape[-1]
    a = -20*torch.exp(-0.2*torch.sqrt(x.pow(2.).sum(dim=-1)/d))
    b = -torch.exp(torch.cos(2*np.pi*x).sum(dim=-1)/d)
    return a + b + np.e + 20
def myackley(x):
    return ackley(x)+torch.sqrt(x.norm(dim=-1))
def rastrigin(x):
    d = x.shape[-1]
    ans = (x.pow(2.)-10*torch.cos(2*np.pi*x)).sum(dim=-1)
    return 10*d + ans
def sphere(x):
    d = x.shape[-1]
    return x.pow(2.).sum(dim=-1)

def rosenbrock(x):
    d = x.shape[-1]
    ans = 100*(x[..., 1:] - (x[..., :-1]).pow(2.)).pow(2.)
    ans = ans+ (1-x[..., :-1]).pow(2.)
    return ans.sum(dim=-1)

def fit_fn(x, optim_fn):
    if type(x) is list:
        x = torch.stack(x)
    fitdata = xr.DataArray(np.zeros((len(x), 2)), dims=['x', 'metric'], 
                           coords={'metric':['fn_val', 'fitness']})
    val = optim_fn(x).detach().cpu().numpy()
    fitdata[:, 0] = val
    fitdata[:, 1] = -val
    return fitdata
    
def lin_fn(x):
    lin_coefs = torch.linspace(-10, 10, 3000).to(x)
    d = x.shape[-1]
    return (lin_coefs[:d]*x).sum(dim=-1)

