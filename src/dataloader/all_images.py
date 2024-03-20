import os
import copy
from os import PathLike
from glob import glob
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.utils.misc import split_path


class AllImageDataset(Dataset):
    def __init__(
        self,
        root_dir: PathLike,
        label_to_search_string: dict[str, str],
        transform: Optional[transforms.Compose] = None,
        device: torch.device = torch.device('cpu')
    ) -> None:
        assert 'default' not in label_to_search_string, "The label 'default' is reserved for leftover images."
        
        all_paths = set(glob('**/*.jpg', root_dir=root_dir, recursive=True))
        all_paths |= set(glob('**/*.png', root_dir=root_dir, recursive=True))
        default_set = all_paths.copy()
        label_to_set: dict[str, set[PathLike]] = dict()
        for label, pattern in label_to_search_string.items():
            label_set = set(glob(pattern, root_dir=root_dir, recursive=True))
            default_set -= label_set
            label_to_set[label] = label_set
        label_to_set['default'] = default_set

        self.annotations = pd.DataFrame(columns=['image', 'path', 'label'])

        for label, label_set in label_to_set.items():
            for path in label_set:
                _, image_id, _ = split_path(path)
                self.annotations.loc[len(self.annotations)] = [image_id, path, label]
        
        self.root_dir = root_dir
        if transform is None:
            transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])
        self.transform = transform
        self.label_to_index = {label: i for i, label in enumerate(np.unique(self.annotations['label']))}
        self.index_to_label = list(self.label_to_index.keys())
        self.device = device

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor]:
        _, relative_path, label = self.annotations.iloc[index]
        path = os.path.join(self.root_dir, relative_path)
        image = Image.open(path).convert('RGB')
        image = self.transform(image).to(self.device)
        label = torch.tensor(self.label_to_index[label]).to(self.device)
        return (image, label)

    def subset(self, labels: list[str]) -> 'AllImageDataset':
        new_dataset = copy.deepcopy(self)
        new_dataset.annotations = new_dataset.annotations[new_dataset.annotations['label'].isin(labels)].reset_index(drop=True)
        return new_dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.utils.config import Config

    cfg = Config('configs/config.yaml')
    data_folder = cfg.get('data', 'raw_path')

    label_to_search_string = dict(never='**/never/**')

    dataset = AllImageDataset(root_dir=data_folder, label_to_search_string=label_to_search_string)

    num_rows = len(dataset.label_to_index)
    fig, axs = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
    for label, idx in dataset.label_to_index.items():
        subset_dataset = dataset.subset([label])
        dataloader = DataLoader(subset_dataset, batch_size=5, shuffle=True)

        images, labels = next(iter(dataloader))
        for i, ax in enumerate(axs[idx]):
            ax.imshow(images[i].permute(1, 2, 0))
            ax.set_title(f'Label: {dataset.index_to_label[labels[i]]}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()
