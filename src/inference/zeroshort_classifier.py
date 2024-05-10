from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.models.CLIP import ZeroShotHead, CLIPWithHead
from src.utils.config import Config
from src.dataloader.annotated_image import AnnotatedImageDataset
from src.dataloader.encodings import EncodedDataset


classes = np.array([
    # 'A picture of an abstract painting',
    'A picture of a painting of a landscape',
    'A picture of a painting of a person',
    'A picture of a painting of an animal'
])

root_num_imgs = 6


cfg = Config('configs/config.yaml')
cropped_dir = Path(cfg.get('data', 'cropped_path'))
encoding_dir = Path(cfg.get('data', 'encoded_path'))
annotations_dir = Path(cfg.get('data', 'elo_annotations_path'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if True:
    dataset = AnnotatedImageDataset.all_image_dataset(
        csv_file=annotations_dir / 'corners.csv',
        device=device
    )
    dataset.annotations = dataset.annotations.drop(columns=['corners'])[dataset.annotations['discard'] == False]
    dataset.root_dir = cropped_dir
    model = CLIPWithHead.with_zeroshot_head(classes.tolist()).to(device).eval()
else:
    dataset = EncodedDataset(
        csv_file=annotations_dir / 'corners.csv',
        root_dir=encoding_dir,
        device=device
    )
    dataset.annotations = dataset.annotations.drop(columns=['corners'])[dataset.annotations['discard'] == False]
    model = ZeroShotHead(classes.tolist()).to(device).eval()


dataloader = DataLoader(dataset, batch_size=root_num_imgs**2)
fig_size_multiplier = 10

with torch.no_grad():
    for i, (x, _) in enumerate(dataloader):
    
        output: torch.Tensor = model(x)
        probs = output.softmax(dim=-1)
        max_prob_idx = probs.argmax(dim=-1)
        max_prob = probs.max(dim=-1).values

        predictions = classes[max_prob_idx]
        labels = predictions
        labels = max_prob_idx
        labels_and_probs = [f'{label} ({prob:.2f})' for label, prob in zip(labels, max_prob)]

        fig, axs = plt.subplots(root_num_imgs, root_num_imgs, figsize=(fig_size_multiplier*root_num_imgs, fig_size_multiplier*root_num_imgs))
        for label, img, ax in zip(labels_and_probs, x, axs.flatten()):
            ax.imshow(img.permute(1, 2, 0).cpu())
            ax.set_title(label)
            ax.axis('off')

        # fig.tight_layout()
        plt.show()
        break
