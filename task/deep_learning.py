import os
import torch
from torch import optim
from model import MinistClassfier, LeNet5, TextCNN
from dataset import CifarDataset, MinistDataset, TextDataset
from dataset.TextDataset import tokenizer, get_word2id, get_relation, get_max_length
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


def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    if labels[0] != "<unkonwn>":
        labels = torch.LongTensor(labels)
    inputs = dict()
    for k in data[0]:
        inputs[k] = list()
    for item in data:
        for k in item:
            inputs[k].append(item[k])
    for k in inputs:
        inputs[k] = torch.LongTensor(inputs[k])
    return [inputs, labels]


class TextClassify:
    def __init__(self, config):

        word2id = get_word2id(
            [
                os.path.join(config.data_root, "data_train.txt"),
                os.path.join(config.data_root, "data_val.txt"),
                os.path.join(config.data_root, "test_exp3.txt"),
            ]
        )
        max_length = get_max_length(
            [
                os.path.join(config.data_root, "data_train.txt"),
                os.path.join(config.data_root, "data_val.txt"),
                os.path.join(config.data_root, "test_exp3.txt"),
            ]
        )
        relation = get_relation(os.path.join(config.data_root, "rel2id.json"))

        self.train_dataset = TextDataset(
            file_path=os.path.join(config.data_root, "data_train.txt"),
            tokenizer=tokenizer,
            word2id=word2id,
            relation=relation,
            max_sent=max_length,
        )
        self.valid_dataset = TextDataset(
            file_path=os.path.join(config.data_root, "data_val.txt"),
            tokenizer=tokenizer,
            word2id=word2id,
            relation=relation,
            max_sent=max_length,
        )
        self.pred_dataset = TextDataset(
            file_path=os.path.join(config.data_root, "test_exp3.txt"),
            tokenizer=tokenizer,
            word2id=word2id,
            relation=relation,
            max_sent=max_length,
            is_test=True,
        )
        device = torch.device("cuda" if config.gpu is not None else "cpu")
        self.model = TextCNN(
            word_dim=10, pos_num=max_length, pos_dim=1, word2id=word2id
        )
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), config.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.collate_fn = collate_fn
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
