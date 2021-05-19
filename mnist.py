import torch
from torch import nn
import torchvision

from tqdm import tqdm

# batch_size_train = 1000
batch_size_train = 64
batch_size_test = 3000
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./mnist/', train=True, download=True,
                             transform=transform),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=False, download=True,
                             transform=transform),
    batch_size=batch_size_test, shuffle=True)

loss_func = nn.NLLLoss()

def perform_stats(net, loader=test_loader, show_stats=True, n_batches=-1, tqdm=tqdm):
    n_correct, total = 0, 0
    loss_total = 0
    n_examples = 0
    loop = enumerate(loader)
    if tqdm is not None:
        loop = tqdm(loop, leave=False, total=max(n_batches, len(loader)))
    for batch_idx, (X_batch, Y_batch) in loop:
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