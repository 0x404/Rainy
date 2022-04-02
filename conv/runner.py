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
import task

logger = get_logger("Runner")


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


class Runner:
    def __init__(self, config):
        self.task = config.task
        self.model = getattr(task, config.task, None)
        if self.model is None:
            raise ValueError(f"task {config.task} is not supported!")


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
                best_accuracy = eval(
                    model, config, eval_dataset, best_accuracy, is_training=True
                )

            if (
                completed_step % config.get("log_every_n_step", 100) == 0
                and completed_step > 0
            ) or completed_step == total_step - 1:
                logger.info(
                    f"[{100 * completed_step / total_step:.2f}%]\tepoch:{epoch}\tstep:{completed_step}\tloss:{loss.item()}"
                )

            if completed_step >= total_step:
                logger.info(
                    f"reach max training step {total_step}, breaking from training loop."
                )
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
    logger.info(
        f"{eval_type} Finish!\tAccuracy:{accuracy * 100:.2f}%\tBest Accuracy:{best_accuracy*100:.2f}%"
    )
    return best_accuracy


if __name__ == "__main__":

    dev_set = CifarDataset(
        "Dataset/validset.txt",
        item_transform=[
            transforms.Normalize(mean=(0.5, 0, 5, 0, 5), std=(0, 5, 0, 5, 0, 5))
        ],
    )
    train_set = CifarDataset(
        "Dataset/trainset.txt",
        item_transform=[
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.5, 0, 5, 0, 5), std=(0, 5, 0, 5, 0, 5)),
        ],
    )
    test_set = CifarDataset("Dataset/trainset.txt", get_label=False)

    net = Net()
    train(net, config, train_set, dev_set)
