import os

import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.models.CLIP import CLIPWithHead
from src.utils.config import Config
from src.dataloader.all_images import AllImageDataset
from src.utils.misc import image_path_to_endocing_path


@torch.no_grad()
def main():
    cfg = Config('configs/config.yaml')
    data_folder = cfg.get('data', 'raw_path')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPWithHead()#.to(device)
    encode_func = lambda images: model.encode(images).cpu().numpy()
    encode_func = lambda images: torch.randn(images.shape[0], 512, device=device).cpu().numpy()
    
    dataset = AllImageDataset(data_folder, use_relative_path_as_label=True, device=device)
    data_loader = DataLoader(dataset, batch_size=32)

    for images, paths in tqdm.tqdm(data_loader):
        encodings = encode_func(images)
        for rel_path, encoding in zip(paths, encodings):
            encoding_path = image_path_to_endocing_path(rel_path)
            path = os.path.join(data_folder, encoding_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, encoding)


if __name__ == "__main__":
    main()