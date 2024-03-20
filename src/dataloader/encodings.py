import os
from os import PathLike
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.misc import image_path_to_endocing_path


class EncodedDataset(Dataset):
    def __init__(
        self,
        csv_file: PathLike,
        root_dir: PathLike,
        device: torch.device = torch.device('cpu')
    ) -> None:
        
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.label_to_index = {label: i for i, label in enumerate(np.unique(self.annotations['label']))}
        self.index_to_label = list(self.label_to_index.keys())
        self.device = device

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor]:
        _, relative_path, label = self.annotations.iloc[index]
        relative_path = image_path_to_endocing_path(relative_path)
        path = os.path.join(self.root_dir, relative_path)
        encoding = np.load(path)
        encoding = torch.from_numpy(encoding).to(self.device)
        label = torch.tensor(self.label_to_index[label]).to(self.device)
        return (encoding, label)


if __name__ == "__main__":
    from src.utils.config import Config

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
    dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

    images, labels = next(iter(dataloader))

    print(images.shape)
    print(labels.shape)
