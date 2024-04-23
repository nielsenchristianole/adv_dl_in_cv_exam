import os

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.models.CLIP import CLIPWithHead, LinearHead
from src.dataloader.annotated_image import AnnotatedImageDataset

csv_file = 'wikiart_test.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = Config('configs/config.yaml')


annotation_path = CFG.get('data', 'annotations_path')
root_dir = CFG.get('data', 'wikiart_path')

dataset = AnnotatedImageDataset(
    csv_file=os.path.join(annotation_path, csv_file),
    root_dir=root_dir,
    device=device
)

dataloader = DataLoader(dataset, shuffle=True)

img: torch.Tensor
label: torch.Tensor
img, label = next(iter(dataloader))

head = LinearHead(dataset.index_to_label)
head.load_state_dict(torch.load('models/head_best.pth', map_location=device))

model = CLIPWithHead(head).to(device).eval()


m = 10
step_size = 0.1
target = torch.tensor((5,), device=device)


delta = torch.zeros_like(img)
for _ in range(m):
    delta = delta.detach().clone().requires_grad_(True)

    x = img + delta

    logits = model(x)
    loss = cross_entropy(logits, target)
    loss.backward()
    print(loss.item())

    delta = delta - step_size * delta.grad


