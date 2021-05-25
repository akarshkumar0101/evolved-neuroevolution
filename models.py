import torch
import torch.nn as nn

class BigConvNet(nn.Module):
    def __init__(self, config=None):
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
        return x.softmax(dim=-1)

class ConvNet(nn.Module):
    def __init__(self, config=None):
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
        return x.softmax(dim=-1)
    
class SmallNet(nn.Module):
    def __init__(self, config=None):
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
        return x.softmax(dim=-1)#, y, x