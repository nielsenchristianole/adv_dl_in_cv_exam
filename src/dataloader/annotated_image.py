import os
from os import PathLike
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.dataloader.base_class import PaintingDataset
from src.utils.config import Config

class AnnotatedImageDataset(PaintingDataset):
    def __init__(
        self,
        csv_file: PathLike,
        root_dir: PathLike,
        transform: Optional[transforms.Compose] = None,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super().__init__(csv_file, root_dir, device)
        
        if transform is None:
            transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])
        self.transform = transform
        self.use_relative_path_as_label = False

    def get_x(self, index: int) -> torch.Tensor:
        _, relative_path, _, _ = self.annotations.iloc[index]
        path = os.path.join(self.root_dir, relative_path)
        image = Image.open(path).convert('RGB')
        image = self.transform(image).to(self.device)
        return image
    
    def get_y(self, index: int) -> tuple[torch.Tensor, str]:
        """How to read the target"""
        img_id, relative_path, label, _hash = self.annotations.iloc[index]
        if self.use_relative_path_as_label:
            return relative_path
        else:
            return torch.tensor(self.label_to_index[label]).to(self.device)
    
    def swap_labels_to_paths(self) -> 'AnnotatedImageDataset':
        self.use_relative_path_as_label = True
        self.index_to_label = None
        self.label_to_index = None
        return self

    @classmethod
    def all_image_dataset(
        cls,
        csv_file: Optional[str]=None,
        splits: Optional[list[str]]=None,
        transform: Optional[transforms.Compose] = None,
        device: torch.device = torch.device('cpu'),
        exclude_never: bool = True
    ) -> 'AnnotatedImageDataset':
        cfg = Config('configs/config.yaml')

        if csv_file is None:
            csv_name = 'all_wanted_images.csv'
            csv_file = os.path.join(cfg.get('data', 'annotations_path'), csv_name)
            if not os.path.exists(csv_file):
                # This will create the csv file
                from src.image_annotation.create_csv_for_all_annotations import main as create_all_wanted_images_csv
                create_all_wanted_images_csv()
        
        this = cls(
            csv_file=csv_file,
            root_dir=cfg.get('data', 'raw_path'),
            transform=transform,
            device=device
        )
        if splits is not None:
            this = this.get_dataset_split_subset(splits)
        this.use_relative_path_as_label = True
        this.index_to_label = None
        this.label_to_index = None
        
        if exclude_never:
            this.annotations = this.annotations[this.annotations['path'].apply(lambda x: 'never' not in x)].reset_index(drop=True)

        return this


if __name__ == "__main__":
    import matplotlib.pyplot as plt
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

    for dataset in (
        AnnotatedImageDataset(csv_file=csv_path, root_dir=data_folder),
        AnnotatedImageDataset.all_image_dataset()
    ):
        dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

        images, labels = next(iter(dataloader))

        # Plot the images in a 3x3 grid
        fig, axs = plt.subplots(3, 3, figsize=(9, 9))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(images[i].permute(1, 2, 0))
            label = dataset.index_to_label[labels[i].item()] if dataset.index_to_label is not None else labels[i]
            ax.set_title(f'Label: {label}')
            ax.axis('off')

        plt.suptitle(f"Dataset length: {len(dataset)}")
        plt.tight_layout()
        plt.show()

