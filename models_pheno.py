import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

class MassiveConvNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.fc1 = nn.Linear(16*7*7, 10)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*7*7)
        x = self.fc1(x)
        return x

class BigConvNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 10)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3)
        self.conv2 = nn.Conv2d(2, 5, kernel_size=3)
        self.conv3 = nn.Conv2d(5, 10, kernel_size=3)
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 10)
        x = self.fc1(x)
        return x
    
class SmallNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3)
        self.fc1 = nn.Linear(9, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 9)
        y = x
        x = self.fc1(x)
        return x

class SmallNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3),
            nn.ReLU(),
            Rearrange('b c y x -> b (c y x)'),
            nn.Linear(9, 10),
        )

    def forward(self, x):
        x = self.seq(x)
        return x