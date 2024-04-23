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

dataloader = DataLoader(dataset, shuffle=False)

img: torch.Tensor
label: torch.Tensor
img, label = next(iter(dataloader))

import matplotlib.pyplot as plt
plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
plt.show()

head = LinearHead(dataset.index_to_label)
head.load_state_dict(torch.load('models/head_best.pth', map_location=device))

model = CLIPWithHead(head).to(device).eval()
model_2 = CLIPWithHead(head, use_shit=True).to(device).eval()

out_1 = model(img)
out_2 = model_2(img)

print(out_1)
print(out_2)

# print(torch.abs(out_1 - out_2).max())
# print(torch.isclose(out_1, out_2).all())

quit()

x = torch.clone(img)
x.requires_grad_(True)

out = model(x)
loss = cross_entropy(out, label)
loss.backward()

grad = x.grad

print(grad)

