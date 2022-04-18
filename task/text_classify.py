import os
import torch
from model import TextCNN
from dataset import TextDataset
from dataset.TextDataset import tokenizer, get_word2id, get_relation, get_max_length
from torch import optim


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
            word_dim=100, pos_num=max_length, pos_dim=5, word2id=word2id
        )
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), config.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
        self.collate_fn = collate_fn
