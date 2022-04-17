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
        self.word2id = word2id
        self.embedding_word = nn.Embedding(
            num_embeddings=len(word2id), embedding_dim=word_dim
        )
        self.embedding1 = nn.Embedding(num_embeddings=pos_num, embedding_dim=pos_dim)
        self.embedding2 = nn.Embedding(num_embeddings=pos_num, embedding_dim=pos_dim)
        self.conv1 = nn.Conv2d(1, 1, (2, 2 * pos_dim + word_dim))
        self.conv2 = nn.Conv2d(1, 1, (3, 2 * pos_dim + word_dim))
        self.conv3 = nn.Conv2d(1, 1, (4, 2 * pos_dim + word_dim))
        self.maxpool1 = nn.MaxPool2d(((pos_num - 2 + 1) // 3, 1))
        self.maxpool2 = nn.MaxPool2d(((pos_num - 3 + 1) // 3, 1))
        self.maxpool3 = nn.MaxPool2d(((pos_num - 4 + 1) // 3, 1))
        self.linear = nn.Linear(9, 44)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        """forward"""
        text = self.embedding_word(data["text"])  # B * pos_num * D
        pos1 = self.embedding1(data["pos1"])  # B * pos_num * pos_dim
        pos2 = self.embedding2(data["pos2"])  # B * pos_num * pos_dim
        input = torch.cat((text, pos1, pos2), dim=2)  # B * pos_num * (D + 2 * pos_dim)
        input = torch.unsqueeze(input, dim=1)  # B * 1 * pos_num * (D + 2 * pos_dim)

        state1 = F.relu(self.conv1(input))  # B * 1 * pos_num - 1 * 1
        state2 = F.relu(self.conv2(input))  # B * 1 * pos_num - 2 * 1
        state3 = F.relu(self.conv3(input))  # B * 1 * pos_num - 3 * 1

        state1 = self.maxpool1(state1).view(-1, 3)  # B * 1 * 3 * 1 -> B * 3
        state2 = self.maxpool2(state2).view(-1, 3)  # B * 1 * 3 * 1 -> B * 3
        state3 = self.maxpool3(state3).view(-1, 3)  # B * 1 * 3 * 1 -> B * 3

        state = torch.cat((state1, state2, state3), dim=1)  # B * 9
        state = self.dropout(state)  # B * 9
        output = self.linear(state)
        return output
