import os
import torch
from torch import optim
from model import MinistClassfier, LeNet5
from dataset import CifarDataset, MinistDataset
import torchvision.transforms as transforms


class MinistClassify:
    def __init__(self, config):
        self.model = MinistClassfier()
        device = torch.device("cuda" if config.gpu is not None else "cpu")
        self.model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), config.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.train_dataset = MinistDataset(
            "train",
            img_dir=os.path.join(config.data_root, "train", "images"),
            label_dir=os.path.join(config.data_root, "train", "labels_train.txt"),
            item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
            label_trans=lambda x: torch.tensor(x, dtype=torch.long),
        )
        self.valid_dataset = MinistDataset(
            "val",
            img_dir=os.path.join(config.data_root, "val", "images"),
            label_dir=os.path.join(config.data_root, "val", "labels_val.txt"),
            item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
            label_trans=lambda x: torch.tensor(x, dtype=torch.long),
        )
        self.pred_dataset = MinistDataset(
            "test",
            img_dir=os.path.join(config.data_root, "test", "images"),
            item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
        )


class ImageClassify:
    def __init__(self, config):
        self.model = LeNet5()
        device = torch.device("cuda" if config.gpu is not None else "cpu")
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), config.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.train_dataset = CifarDataset(
            file=os.path.join(config.data_root, "trainset.txt"),
            data_base_dir=os.path.join(config.data_root, "image"),
            item_transform=[
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=(0.5, 0, 5, 0, 5), std=(0, 5, 0, 5, 0, 5)),
            ],
        )
        self.valid_dataset = CifarDataset(
            file=os.path.join(config.data_root, "validset.txt"),
            data_base_dir=os.path.join(config.data_root, "image"),
            item_transform=[
                transforms.Normalize(mean=(0.5, 0, 5, 0, 5), std=(0, 5, 0, 5, 0, 5))
            ],
        )
        self.pred_dataset = CifarDataset(
            file=os.path.join(config.data_root, "testset.txt"),
            data_base_dir=os.path.join(config.data_root, "image"),
            get_label=False,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
