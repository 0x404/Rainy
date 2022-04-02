import torch
from torch import optim
from model import MinistClassfier, LeNet5


class MinistClassify:
    def __init__(self, config):
        self.model = MinistClassfier()
        self.optimizer = optim.SGD(self.model.parameters(), config.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()


class ImageClassify:
    def __init__(self, config):
        self.model = LeNet5()
        self.optimizer = optim.Adam(self.model.parameters(), config.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
