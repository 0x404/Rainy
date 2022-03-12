"""
author: 0x404
Date: 2022-03-09 19:23:16
LastEditTime: 2022-03-09 23:49:26
Description: 
"""
import os
import torch
from tqdm import tqdm
from imageio import imread
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

config = {
    "batch_size": 64,
    "max_epoch": 10,
    "max_iteration": 1000,
    "learning_rate": 1e-3,
}


class MinistDataset(Dataset):
    def __init__(self, mode, img_dir, label_dir, item_trans=None, label_trans=None):
        super().__init__()
        self.lables = []
        self.mode = mode
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.item_transform = item_trans
        self.label_transform = label_trans
        self._load_lables()

    def _load_lables(self):
        self.lables = []
        with open(self.label_dir, mode="r") as f:
            for _, label in enumerate(f):
                self.lables.append(int(label))

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{self.mode}_{index}.bmp")
        image = imread(img_path)
        if self.item_transform is not None:
            image = self.item_transform(image)
        label = self.lables[index]
        if self.label_transform is not None:
            label = self.label_transform(label)
        return image, label


train_dataset = MinistDataset(
    "train",
    "dataset/train/images",
    "dataset/train/labels_train.txt",
    item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
    label_trans=lambda x: torch.tensor(x, dtype=torch.long),
)
valid_dataset = MinistDataset(
    "val",
    "dataset/val/images",
    "dataset/val/labels_val.txt",
    item_trans=lambda x: torch.tensor(x.flatten(), dtype=torch.float32),
    label_trans=lambda x: torch.tensor(x, dtype=torch.long),
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.get("batch_size", 32), shuffle=True
)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.get("batch_size", 32))


class MinistClassfier(nn.Module):
    def __init__(self):
        super(MinistClassfier, self).__init__()
        hidden_num = config.get("hidden_num", 512)
        self.linear1 = nn.Linear(784, hidden_num)
        self.linear2 = nn.Linear(hidden_num, hidden_num)
        self.linear3 = nn.Linear(hidden_num, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input = self.linear1(input)
        input = self.relu(input)
        input = self.linear2(input)
        input = self.relu(input)
        input = self.linear3(input)
        input = self.softmax(input)
        return input


classfier = MinistClassfier()


for p in classfier.parameters():
    print(p)
for p in classfier.linear2.parameters():
    print(p)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classfier.parameters(), config.get("learning_rate", 0.001))

def train(epoch_num):

    for index, data in enumerate(train_dataloader, 0):
        images, true_labels = data
        pred = classfier(images)
        loss = loss_fn(pred, true_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print(f"epoch[{epoch_num}] step[{index}] loss = {loss.item()}")


def validation(best_accuracy=0.0):
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():
        for data in tqdm(valid_dataloader, desc="validation"):
            images, true_labels = data
            pred = classfier(images)
            total += pred.shape[0]
            correct += sum(
                [(pred[i].argmax(0) == true_labels[i]) for i in range(pred.shape[0])]
            ).item()

    accuracy = correct / total
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(classfier, 'minist-classfier-best-10epoch')
    return best_accuracy
    

if __name__ == "__main__":
    best_accuracy = 0.0
    for epoch in range(config.get("max_epoch")):
        train(epoch)
        best_accuracy = validation(best_accuracy)
