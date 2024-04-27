import os
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import Config
from src.dataloader.base_class import PaintingDataset


def load_whole_dataset(
    csv_name: Optional[str],
    device: torch.device=torch.device('cpu'),
    splits: list[str]=['train']) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = Config('configs/config.yaml')

    csv = os.path.join(cfg.get('data', 'annotations_path'), csv_name or 'all_wanted_images.csv')
    dataset = EncodedDataset(csv, cfg.get('data', 'raw_path')).get_dataset_split_subset(splits)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    encodings = list()
    labels = list()
    for e, l in dataloader:
        encodings.append(e)
        labels.append(l)
    encodings = torch.concat(encodings, dim=0)
    labels = torch.concat(labels, dim=0)
    return encodings.to(device), labels.to(device)


class EncodedDataset(PaintingDataset):
    def __init__(
        self,
        csv_file: PathLike,
        root_dir: PathLike,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super().__init__(csv_file, root_dir, device)

    def get_x(self, index: int) -> torch.Tensor:
        _, relative_path, _, _ = self.annotations.iloc[index]
        relative_path = Path(relative_path).with_suffix('.npy')
        path = os.path.join(self.root_dir, relative_path)
        encoding = np.load(path)
        encoding = torch.from_numpy(encoding).to(self.device)
        return encoding


if __name__ == "__main__":

    cfg = Config('configs/config.yaml')
    data_folder = cfg.get('data', 'encoded_path')
    ann_folder = cfg.get('data', 'annotations_path')

    csv_path = os.path.join(ann_folder, 'calle2.csv')

    dataset = EncodedDataset(csv_file=csv_path, root_dir=data_folder)
    partial_dataset = dataset.get_dataset_split_subset('train')
    dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

    images, labels = next(iter(dataloader))

    print(images.shape)
    print(labels.shape)
