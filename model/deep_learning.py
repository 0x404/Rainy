import torch
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


class TextCNN(nn.Module):
    def __init__(self, word_dim, pos_num, pos_dim, word2id):
        super(TextCNN, self).__init__()
        self.feature_num = 3
        self.feature_dim = word_dim + 2 * pos_dim
        self.kernel_size = [2, 3, 5, 7, 9, 11]
        self.kernel_num = len(self.kernel_size)

        self.word_embedding = nn.Embedding(len(word2id), embedding_dim=word_dim)
        self.pos1_embedding = nn.Embedding(pos_num, embedding_dim=pos_dim)
        self.pos2_embedding = nn.Embedding(pos_num, embedding_dim=pos_dim)
        self.convs = nn.ModuleList(
            nn.Conv2d(1, 1, (i, self.feature_dim)) for i in self.kernel_size
        )
        self.maxpools = nn.ModuleList(
            nn.MaxPool2d(((pos_num - i + 1) // self.feature_num, 1))
            for i in self.kernel_size
        )
        self.linear = nn.Linear(self.kernel_num * self.feature_num, 44)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        """forward"""
        text = self.word_embedding(data["text"])
        pos1 = self.pos1_embedding(data["pos1"])
        pos2 = self.pos2_embedding(data["pos2"])
        input = torch.cat((text, pos1, pos2), dim=2)
        input = torch.unsqueeze(input, dim=1)

        states = [F.relu(conv(input)) for conv in self.convs]
        states_pooled = [maxpool(states[i]) for i, maxpool in enumerate(self.maxpools)]
        states_pooled = [state.view(-1, 3) for state in states_pooled]
        states_pooled = torch.cat(states_pooled, dim=1)
        states_pooled = self.dropout(states_pooled)
        output = self.linear(states_pooled)
        return output
