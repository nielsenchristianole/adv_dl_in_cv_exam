import os
from pathlib import Path
from typing import Optional

import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.models.CLIP import CLIPWithHead
from src.utils.config import Config
from src.dataloader.annotated_image import AnnotatedImageDataset


@torch.no_grad()
def main(device: Optional[torch.device]=None):
    cfg = Config('configs/config.yaml')
    cropped_folder = Path(cfg.get('data', 'cropped_path'))
    encoding_folder = Path(cfg.get('data', 'encoded_path'))

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPWithHead.with_classification_head('class').to(device).eval()
    
    dataset = AnnotatedImageDataset.all_image_dataset(
        csv_file=Path(cfg.get('data', 'elo_annotations_path')) / 'corners.csv',
        device=device
    )
    dataset.root_dir = cropped_folder
    dataset.annotations = dataset.annotations.drop(columns=['corners'])[dataset.annotations['discard'] == False]

    data_loader = DataLoader(dataset, batch_size=32)

    for images, paths in tqdm.tqdm(data_loader):
        encodings = model.embed_image(images).cpu().numpy()
        for rel_path, encoding in zip(paths, encodings):
            out_path = encoding_folder / Path(rel_path).with_suffix('.npy')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), encoding)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    main(device)