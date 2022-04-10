import torch.nn.functional as F
from torch import nn


class MinistClassfier(nn.Module):
    """MinistClassfier"""

    def __init__(self, hidden_num=512):
        super(MinistClassfier, self).__init__()
        self.linear1 = nn.Linear(784, hidden_num)
        self.linear2 = nn.Linear(hidden_num, hidden_num)
        self.linear3 = nn.Linear(hidden_num, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        """Forward function"""
        input = self.linear1(input)
        input = self.relu(input)
        input = self.linear2(input)
        input = self.relu(input)
        input = self.linear3(input)
        input = self.softmax(input)
        return input


class LeNet5(nn.Module):
    """LeNet5"""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (5, 5))
        self.conv2 = nn.Conv2d(32, 24, (7, 7), padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(24, 8, (5, 5), padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.linear1 = nn.Linear(8 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward function"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x).view(-1, 8 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
