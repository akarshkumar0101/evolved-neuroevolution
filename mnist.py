import torch
from torch import nn
import torchvision

from tqdm import tqdm


class MNIST:
    def __init__(self, ):
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

        self.ds_train = torchvision.datasets.MNIST('./', train=True, download=True, transform=self.transform)
        self.loader_train = torch.utils.data.DataLoader(self.ds_train, 
                                                        batch_size=len(self.ds_train), shuffle=True)
        self.ds_test = torchvision.datasets.MNIST('./', train=False, download=True, transform=self.transform)
        self.loader_test = torch.utils.data.DataLoader(self.ds_test, 
                                                       batch_size=len(self.ds_test), shuffle=True)
        
        self.loss_func = nn.NLLLoss()

    def load_all_data(self, device='cpu'):
        self.X_train,self.Y_train = next(iter(self.loader_train))
        self.X_test, self.Y_test = next(iter(self.loader_test))
        self.X_train,self.Y_train = self.X_train.to(device), self.Y_train.to(device)
        self.X_test, self.Y_test = self.X_test.to(device), self.Y_test.to(device)


    def perform_stats(self, net, loader=None, show_stats=True, n_batches=-1, tqdm=tqdm, device='cpu'):
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
            loss = loss_func(Y_batch_pred.log(), Y_batch).item()
            loss_total += loss * len(X_batch)
            n_examples += len(X_batch)
            total += len(Y_batch)
        loss_total /= n_examples
        accuracy = n_correct/total
        if show_stats:
            print(f'Average Loss: {loss_total:.03f}, Accuracy: {accuracy*100:.03f}%')
        return loss_total, accuracy


    def calc_pheo_fitness(self, pheno, n_sample=1000, device='cpu'):
        idx = torch.randperm(len(self.X_train))[:n_sample]
        X_batch, Y_batch = self.X_train[idx].to(device), self.Y_train[idx].to(device)

        Y_batch_pred = pheno(X_batch)
        n_correct = (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()
        loss = self.loss_func(Y_batch_pred.log(), Y_batch).item()
    #     if not np.isfinite(loss):
    #         loss = 10.
        accuracy = n_correct/len(Y_batch)
        return {'fitness': -loss, 'loss': loss, 'accuracy': accuracy}

