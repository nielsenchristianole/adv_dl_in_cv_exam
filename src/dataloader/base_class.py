import os
from os import PathLike
from typing import Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import Config


class PaintingDataset(Dataset, ABC):
    def __init__(
        self,
        csv_file: PathLike,
        root_dir: PathLike,
        device: torch.device = torch.device('cpu')
    ) -> None:
        self.cfg = Config('configs/config.yaml')
        
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.use_labels = 'label' in self.annotations.columns

        if self.use_labels:
            self.label_to_index = {label: i for i, label in enumerate(np.unique(self.annotations['label']))}
            self.index_to_label = list(self.label_to_index.keys())
        self.device = device

    def __len__(self) -> int:
        return len(self.annotations)
    
    @abstractmethod
    def get_x(self, index: int) -> Any:
        """How to read the input"""
    
    def get_y(self, index: int) -> Any:
        """How to read the target"""
        if not self.use_labels:
            return torch.tensor(0).to(self.device)
        img_id, relative_path, label, _hash = self.annotations.iloc[index]
        label = torch.tensor(self.label_to_index[label]).to(self.device)
        return label

    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor]:
        x = self.get_x(index)
        y = self.get_y(index)
        return (x, y)
    
    def get_split_from_hash(self, _hash: float) -> str:
        """Get the split from the hash"""
        train_split = self.cfg.get('data', 'data_split', 'train')
        if _hash < train_split:
            return 'train'
        elif _hash < train_split + self.cfg.get('data', 'data_split', 'val'):
            return 'val'
        else:
            return 'test'
        
    def get_dataset_split_subset(self, splits: list[str]|str) -> 'PaintingDataset':
        """Get a subset of the dataset based on data splits"""
        if isinstance(splits, str):
            splits = [splits]
        assert all(split in ('train', 'val', 'test') for split in splits), f"Invalid split in {splits=}"
        
        dataset_splits = self.annotations['hash'].apply(self.get_split_from_hash)
        mask = dataset_splits.isin(splits)

        this = deepcopy(self)
        this.annotations = this.annotations[mask].reset_index(drop=True)
        return this

    def get_dataset_label_subset(self, labels: list[str]|str) -> 'PaintingDataset':
        """Get a subset of the dataset based on labels"""
        if isinstance(labels, str):
            labels = [labels]
        assert all(label in self.index_to_label for label in labels), f"Invalid label in {labels=}"
        
        mask = self.annotations['label'].isin(labels)

        this = deepcopy(self)
        this.annotations = this.annotations[mask].reset_index(drop=True)
        return this

    def split_on_hash(self, splitter: Callable[[float], bool]) -> 'PaintingDataset':
        """Give a function which takes a float in the [0,1] interval and returns wether ot not is should belong to the dataset"""
        hashes = self.annotations['hash'].tolist()
        mask = np.array([splitter(h) for h in hashes])

        this = deepcopy(self)
        this.annotations = this.annotations[mask].reset_index(drop=True)
        return this
