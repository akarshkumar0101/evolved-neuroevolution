import numpy as np
import torch
from torch import nn
import torchvision

from tqdm import tqdm

from multi_arr import MultiArr
import xarray as xr

class MNIST:
    def __init__(self, ):
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

        self.ds_train = torchvision.datasets.MNIST('~/datasets/mnist', train=True, 
                                                   download=True, transform=self.transform)
        self.ds_test = torchvision.datasets.MNIST('~/datasets/mnist', train=False, 
                                                  download=True, transform=self.transform)
        self.bs_train = 1000
        self.bs_test = 3000
        self.loader_train = torch.utils.data.DataLoader(self.ds_train, 
                                                        batch_size=self.bs_train, shuffle=True)
        self.loader_test = torch.utils.data.DataLoader(self.ds_test, 
                                                       batch_size=self.bs_test, shuffle=True)
        self.loss_func = nn.NLLLoss()

    def load_all_data(self, device='cpu'):
        data = [(Xb, Yb) for Xb, Yb in self.loader_train]
        self.X_train = torch.cat([di[0] for di in data], dim=0).to(device)
        self.Y_train = torch.cat([di[1] for di in data], dim=0).to(device)
        
        data = [(Xb, Yb) for Xb, Yb in self.loader_test]
        self.X_test = torch.cat([di[0] for di in data], dim=0).to(device)
        self.Y_test = torch.cat([di[1] for di in data], dim=0).to(device)

    def perform_stats(self, net, loader=None, verbose=True, n_batches=-1, tqdm=tqdm, device='cpu'):
        if loader is None:
            loader = self.loader_test
        n_correct, total = 0, 0
        loss_total = 0
        n_examples = 0
        loop = enumerate(loader)
        if tqdm is not None:
            loop = tqdm(loop, leave=False, total=max(n_batches, len(loader)))
        for batch_idx, (X_batch, Y_batch) in loop:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            if batch_idx == n_batches:
                break
            Y_batch_pred = net(X_batch)
            n_correct += (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()
            loss = self.loss_func(Y_batch_pred, Y_batch).item()
            loss_total += loss * len(X_batch)
            n_examples += len(X_batch)
            total += len(Y_batch)
        loss_total /= n_examples
        accuracy = n_correct/total*100.
        if verbose:
            print(f'Average Loss: {loss_total:.03f}, Accuracy: {accuracy:.03f}%')
        return {'loss': loss_total, 'accuracy': accuracy}

    # TODO: do not use .item() anywhere and rather just accumulate gpu tensor data.
    # transferring from gpu mem to cpu mem takes FOREVER in clock time
    def calc_pheno_fitness(self, pheno, n_samples=5000, device=None, ds='train'):
        if ds=='train':
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        if n_samples is None:
            idx = torch.arange(len(X))
        else:
            idx = torch.randperm(len(X))[:n_samples]
        
        pheno = pheno.to(device)
            
        X_batch, Y_batch = X[idx].to(device), Y[idx].to(device)

        Y_batch_pred = pheno(X_batch)
        loss = self.loss_func(Y_batch_pred, Y_batch).item()
        n_correct = (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()
        accuracy = n_correct/len(Y_batch)*100.
        return {'fitness': -loss, 'loss': loss, 'accuracy': accuracy}

    def calc_pop_fitness(self, pop, geno2pheno, n_samples=5000, device=None, ds='train'):
        if ds=='train':
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        if n_samples is None:
            idx = torch.arange(len(X))
        else:
            idx = torch.randperm(len(X))[:n_samples]
            idx = torch.arange(n_samples)
        
        X_batch, Y_batch = X[idx].to(device), Y[idx].to(device)

        fitdata = xr.DataArray(np.zeros((len(pop), 3)), dims=['pop', 'metric'], 
                               coords={'metric':['loss', 'fitness', 'accuracy']})
        for i, x in enumerate(pop):
            net = geno2pheno(x).to(device)
            Y_batch_pred = net(X_batch)
            loss = self.loss_func(Y_batch_pred, Y_batch).item()
            n_correct = (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()
            accuracy = n_correct/len(Y_batch)*100.
            fitdata[i, :] = [loss, -loss, accuracy]
        return fitdata
    
