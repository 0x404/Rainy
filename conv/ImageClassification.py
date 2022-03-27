import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils import get_logger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm


logger = get_logger('Trainer')

config = {
    "batch_size": 32,
    "epochs": 100,
    "acc_step": 1,
    "log_every_n_step": 200,
    "lr": 0.00005,
    # "total_step": 50000,
    "save_ckpt_n_step": 2000,
    "T_max": 10,
}


class CifarDataset(Dataset):
    """DataSet for Cifar task.
    
    Args:
        file: data file path.
        data_base_dir: directory of images.
        get_lable: whther this Dataset has labels.
    """

    def __init__(
        self, file, data_base_dir="Dataset/image/", get_label=True, item_transform=None
    ):
        if not os.path.isdir(data_base_dir):
            raise ValueError(f"{data_base_dir} is not a dir!")
        self.item_transform = None
        if item_transform is not None and not isinstance(item_transform, list):
            self.item_transform = [item_transform]
        self.base_dir = data_base_dir
        self.get_label = get_label
        self.file = file
        self.images = list()
        self.labels = list()
        self.from_file(file)

    def from_file(self, file):
        """Construct DataSet from file"""
        self.reset()
        self.file = file
        with open(file, mode="r") as f:
            for line in f:
                line = line.strip("\n")
                if len(line.split(" ")) == 2:
                    image, label = line.split(" ")
                elif not self.get_label:
                    image = line
                    label = 0
                else:
                    logger.error(f"data set {self.file} has no labels!")
                    raise ValueError
                self.images.append(image)
                self.labels.append(int(label))

    def reset(self):
        """Reset DataSet"""
        self.images = list()
        self.labels = list()

    def __getitem__(self, index):
        assert index < len(self.labels)
        try:
            image = read_image(os.path.join(self.base_dir, self.images[index]))
            image = image.to(torch.float32)
            if self.item_transform is not None:
                for t in self.item_transform:
                    image = t(image)
            label = torch.tensor(self.labels[index], dtype=torch.long)
        except:
            logger.error(f"Get error when read item {index} in {self.file}")
            raise RuntimeError()
        return image, label

    def __len__(self):
        return len(self.labels)


class Net(nn.Module):
    """LeNet5"""

    def __init__(self):
        super(Net, self).__init__()
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
        x = self.pool2(x).view(-1, 8 * 5 *5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train(model, config, train_dataset, eval_dataset):
    """Train a model.
    
    Args:
        model: sub-class of torch.nn.Module, the model you want to train.
        config: python dict, consists of hyperparameters.
        train_dataset: sub-class of torch.utils.data.DataSet, your train data set.
        eval_dataset: sub-class of torch.utils.data.DataSet, your eval data set.
    """
    total_step = config.get("total_step", -1)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    if total_step > len(train_loader) * config.get("epochs") or total_step == -1:
        total_step = len(train_loader) * config.get("epochs")

    logger.info("********** Running training **********")
    logger.info(f"  Num Examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {config.get('epochs')}")
    logger.info(f"  Global Total Step = {total_step}")
    logger.info(f"  Accumulate Gradient Step = {config.get('acc_step')}")
    logger.info(f"  Model Structure = {model}")

    best_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["T_max"]
    )
    optimizer.zero_grad()

    completed_step = 0
    for epoch in range(config["epochs"]):
        logger.info(f"epoch : {epoch}\tlearning rate : {lr_scheduler.get_last_lr()}")
        for step, batch in enumerate(train_loader):
            images, labels = batch
            if epoch == 0 and step == 0:
                logger.info(f"images: {images}")
                logger.info(f"images dtype: {images.dtype}")
                logger.info(f"images shape: {images.shape}")
                logger.info(f"labels: {labels}")
                logger.info(f"labels shape: {labels.shape}")
                logger.info(f"labels dtype: {labels.dtype}")

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            if step % config["acc_step"] == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                completed_step += 1

            if (
                completed_step % config.get("save_ckpt_n_step", 5000) == 0
                and completed_step > 0
            ):
                best_accuracy = eval(model, config, eval_dataset, best_accuracy, is_training=True)

            if (
                completed_step % config.get("log_every_n_step", 100) == 0
                and completed_step > 0
            ) or completed_step == total_step - 1:
                logger.info(f"[{100 * completed_step / total_step:.2f}%]\tepoch:{epoch}\tstep:{completed_step}\tloss:{loss.item()}")

            if completed_step >= total_step:
                logger.info(f"reach max training step {total_step}, breaking from training loop.")
                break
        lr_scheduler.step()
    logger.info("Training Finish!")
    eval(model, config, eval_dataset, best_accuracy, is_training=True)


def eval(model, config, eval_dataset, best_accuracy, is_training=False):
    """Evalution a model.
    
    Args:
        model: sub-class of torch.nn.Module, the model you want to evaluate.
        config: python dict, consists of hyperparameters.
        eval_dataset: sub-class of torch.utils.data.DataSet, your eval data set.
        beset_accuracy: best accuracy in history.
        is_training: whther the model been evaluated is training.
    """
    eval_loader = DataLoader(eval_dataset)
    eval_type = "Evalution" if is_training else "Prediction"
    logger.info(f"********** Running {eval_type} **********")
    logger.info(f"  Num Examples = {len(eval_loader)}")
    logger.info(f"  Batch Size = {config['batch_size']}")
    total_items = 0
    correct_items = 0
    for _, batch in enumerate(tqdm(eval_loader, desc=eval_type)):
        with torch.no_grad():
            images, labels = batch
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            batch_size = pred.shape[0]
            correct_items += sum(
                [1 if pred[i] == labels[i] else 0 for i in range(batch_size)]
            )
            total_items += batch_size
    accuracy = correct_items / total_items
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model, "model-best-accuracy")
    logger.info(f"{eval_type} Finish!\tAccuracy:{accuracy * 100:.2f}%\tBest Accuracy:{best_accuracy*100:.2f}%")
    return best_accuracy


if __name__ == "__main__":

    dev_set = CifarDataset(
        "Dataset/validset.txt",
        item_transform=[
            transforms.Normalize(mean=(0.5, 0,5, 0,5), std=(0,5, 0,5, 0,5))
        ],
    )
    train_set = CifarDataset(
        "Dataset/trainset.txt",
        item_transform=[
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.5, 0,5, 0,5), std=(0,5, 0,5, 0,5)),
        ],
    )
    test_set = CifarDataset("Dataset/trainset.txt", get_label=False)

    net = Net()
    train(net, config, train_set, dev_set)
