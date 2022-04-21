"""Dataset for minist"""
import os
from imageio import imread
from torch.utils.data import Dataset


class MinistDataset(Dataset):
    """Dataset for minist"""

    def __init__(
        self, mode, img_dir, label_dir=None, item_trans=None, label_trans=None
    ):
        super().__init__()
        self.lables = []
        self.mode = mode
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.item_transform = item_trans
        self.label_transform = label_trans
        if label_dir is not None:
            self._load_lables()

    def _load_lables(self):
        self.lables = []
        with open(self.label_dir, mode="r") as file:
            for _, label in enumerate(file):
                self.lables.append(int(label))

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{self.mode}_{index}.bmp")
        image = imread(img_path)
        if self.item_transform is not None:
            image = self.item_transform(image)
        if self.label_dir is None:
            label = 0
        else:
            label = self.lables[index]
        if self.label_transform is not None:
            label = self.label_transform(label)
        return image, label
