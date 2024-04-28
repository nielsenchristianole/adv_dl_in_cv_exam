import os

import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.dataloader.annotated_image import AnnotatedImageDataset
from src.models.diffusion_sampling import cond_sample, sample, plot_tensor
from src.models.CLIP import CLIPWithHead, LinearHead
from src.models.diffusion import WikiArt256Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

num_classes = 27
linear_head_model = LinearHead(num_classes * [None])
linear_head_model.load_state_dict(torch.load('./models/head_best.pth'))

classifier = CLIPWithHead(linear_head_model, crop_and_norm=True).to(device).eval()

dataset = AnnotatedImageDataset(
    csv_file='data/annotations/wikiart_test.csv',
    root_dir='data/wikiart',
    device=device)
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

diff_model = WikiArt256Model().to(device)
diff_model.load_state_dict(torch.load('models/wikiart_256.pth', map_location=device))
diff_model.eval()


num_samples = 5
steps = 100
eta=1

with torch.no_grad():
    for i in tqdm.trange(58, int(1000/num_samples)):
        _sample = sample(
            model=diff_model,
            x=torch.randn([num_samples, 3, 256, 256], requires_grad=True).to(device),
            steps=torch.linspace(1, 0, steps + 1)[:-1].to(device),
            eta=eta,
            extra_args=dict()
        )
        if _sample is None:
            continue
        out = classifier(_sample)
        preds = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).tolist()
        for j, (pred, im) in enumerate(zip(preds, _sample)):
            out_dir = f'./data/samples/{dataset.index_to_label[pred]}/'
            os.makedirs(out_dir, exist_ok=True)
            np.save(f'{out_dir}/sample_{i}_{j}.npy', im.detach().cpu().numpy())
