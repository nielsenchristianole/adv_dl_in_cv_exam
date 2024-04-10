import os
from typing import Optional

import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.models.CLIP import CLIPWithHead
from src.utils.config import Config
from src.dataloader.annotated_image import AnnotatedImageDataset
from src.utils.misc import image_path_to_encoding_path


@torch.no_grad()
def main(device: Optional[torch.device]=None):
    cfg = Config('configs/config.yaml')
    data_folder = cfg.get('data', 'raw_path')

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPWithHead.with_zeroshot_head('class').to(device).eval()
    
    dataset = AnnotatedImageDataset.all_image_dataset(device=device)
    data_loader = DataLoader(dataset, batch_size=32)

    for images, paths in tqdm.tqdm(data_loader):
        encodings = model.embed_image(images).cpu().numpy()
        for rel_path, encoding in zip(paths, encodings):
            encoding_path = image_path_to_encoding_path(rel_path)
            path = os.path.join(data_folder, encoding_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, encoding)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    main(device)