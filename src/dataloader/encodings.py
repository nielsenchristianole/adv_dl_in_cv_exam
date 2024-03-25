import os
from os import PathLike
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import Config
from src.utils.misc import image_path_to_encoding_path
from src.dataloader.base_class import PaintingDataset


def load_whole_dataset(csv_name: Optional[str]) -> torch.Tensor:
    cfg = Config('configs/config.yaml')

    csv = os.path.join(cfg.get('data', 'annotations_path'), csv_name or 'all_wanted_images.csv')
    dataset = EncodedDataset(csv, cfg.get('data', 'raw_path'))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    encodings = torch.concat([e for e, l in dataloader])
    return encodings


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
        relative_path = image_path_to_encoding_path(relative_path)
        path = os.path.join(self.root_dir, relative_path)
        encoding = np.load(path)
        encoding = torch.from_numpy(encoding).to(self.device)
        return encoding


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and display images from a CSV file.")
    parser.add_argument('--file', type=str, help="The name of the CSV file without extension.")
    args = parser.parse_args()

    cfg = Config('configs/config.yaml')
    data_folder = cfg.get('data', 'raw_path')
    ann_folder = cfg.get('data', 'annotations_path')

    csv_name = args.file
    if csv_name is None:
        csv_name, _ = os.path.splitext(os.listdir(ann_folder)[0])
    csv_path = os.path.join(ann_folder, f"{csv_name}.csv")

    dataset = EncodedDataset(csv_file=csv_path, root_dir=data_folder)
    partial_dataset = dataset.get_dataset_subset('train')
    dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

    images, labels = next(iter(dataloader))

    print(images.shape)
    print(labels.shape)
