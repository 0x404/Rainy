"""Task for text relation extraction"""
# pylint: disable=too-few-public-methods
import os
import torch
from torch import optim
from model import TextCNN
from dataset import TextDataset
from dataset.text_dataset import tokenizer, get_word2id, get_relation, get_max_length


def collate_fn(batch):
    """collate fuction for this task"""
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = {k: [] for k in data[0]}
    if labels[0] != "<unkonwn>":
        labels = torch.LongTensor(labels)
    for item in data:
        for k in item:
            inputs[k].append(item[k])
    inputs = {k: torch.LongTensor(tensor) for k, tensor in inputs.items()}
    return [inputs, labels]


class RelationExtract:
    """Relation Extract"""

    def __init__(self, config):
        data_root = config.data.data_root
        word2id = get_word2id(
            [
                os.path.join(data_root, "data_train.txt"),
                os.path.join(data_root, "data_val.txt"),
                os.path.join(data_root, "test_exp3.txt"),
            ]
        )
        max_length = get_max_length(
            [
                os.path.join(data_root, "data_train.txt"),
                os.path.join(data_root, "data_val.txt"),
                os.path.join(data_root, "test_exp3.txt"),
            ]
        )
        relation = get_relation(os.path.join(data_root, "rel2id.json"))

        self.train_dataset = TextDataset(
            file_path=os.path.join(data_root, "data_train.txt"),
            tokenizer=tokenizer,
            word2id=word2id,
            relation=relation,
            max_sent=max_length,
        )
        self.valid_dataset = TextDataset(
            file_path=os.path.join(data_root, "data_val.txt"),
            tokenizer=tokenizer,
            word2id=word2id,
            relation=relation,
            max_sent=max_length,
        )
        self.pred_dataset = TextDataset(
            file_path=os.path.join(data_root, "test_exp3.txt"),
            tokenizer=tokenizer,
            word2id=word2id,
            relation=relation,
            max_sent=max_length,
            is_test=True,
        )

        device = torch.device(config.setup.device)
        self.model = TextCNN(
            word_dim=100, pos_num=max_length, pos_dim=5, word2id=word2id
        )
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), config.train.lr)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10
        )
        self.collate_fn = collate_fn
