import os
import logging
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

logger = logging.getLogger(__name__)


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
