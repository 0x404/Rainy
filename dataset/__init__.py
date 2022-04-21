"""All dataset"""
from .cifar_dataset import CifarDataset
from .minist_dataset import MinistDataset
from .text_dataset import TextDataset

__all__ = ["CifarDataset", "MinistDataset", "TextDataset"]
