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

classifier = CLIPWithHead(linear_head_model, crop_and_norm=False).to(device).eval()

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
target_label = 'Cubism'


forward_guidance_scale=23
num_backward_steps=5
backward_step_size=1e-1
backward_guidance_scale=0.8


target_index = dataset.label_to_index[target_label]
target_tensor = torch.tensor(num_samples*(target_index,), device=device)
_sample = cond_sample(
    model=diff_model,
    x=torch.randn([num_samples, 3, 256, 256], requires_grad=True).to(device),
    steps=torch.linspace(1, 0, steps + 1)[:-1].to(device),
    eta=eta,
    classifier=classifier,
    label=target_tensor,
    num_backward_steps=num_backward_steps,
    backward_guidance_scale=backward_guidance_scale,
    forward_guidance_scale=forward_guidance_scale,
    backward_step_size=backward_step_size
)
if _sample is None:
    print("No valid samples found.")
    quit()


with torch.no_grad():
    out = classifier(_sample)
    loss = torch.nn.functional.cross_entropy(out, target_tensor)
    probs = torch.nn.functional.softmax(out, dim=1)
    print(f"Loss: {loss.item()}")
    print(f"Target class: {target_index}, {target_label}")
    print(f"Predicted class: {probs.argmax(dim=1).detach().cpu().numpy()}")
    for i in range(num_samples):
        print(f"Sample {i}: {dataset.index_to_label[probs.argmax(dim=1)[i].detach().cpu().numpy()]}")
    print(f"Target class probability: {probs[:, target_index].detach().cpu().numpy()}")
    print(f"Predicted probability: {probs.max(dim=1).values.detach().cpu().numpy()}")
    # print(np.array2string(probs.detach().cpu().numpy(), precision=3))

    plot_tensor(_sample)