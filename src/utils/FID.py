import os
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import numpy as np
from src.models.diffusion import WikiArt256Model
from src.models.CLIP import CLIPWithHead, LinearHead
import src.models.diffusion_sampling as sampling
from src.dataloader.annotated_image import AnnotatedImageDataset

# import the partials package
from functools import partial
from pathlib import Path

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features()[:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        feat = self.features(x)
        feat = self.avg_pool(feat)
        return feat
        
class FID:
    
    def __init__(self, feature_extractor, device, samples_per_fid = 1000):
        self.device = device
        
        self.feature_extractor = feature_extractor
        self.samples_per_fid = samples_per_fid
        
    def calculate_fid(self, *, dataloader, sample_method, class_idx, class_name):
        # TODO: Add method description

        os.makedirs('out/FID_data', exist_ok=True)
        
        # TODO: Adjust the number of samples by samples_per_fid
        features_real = np.empty((len(dataloader.dataset), 256))
        features_gen = np.empty((len(dataloader.dataset), 256))
        
        start_idx = 0
        for i, (images, _) in tqdm(enumerate(dataloader), leave=False):
            n_samples = images.shape[0]
            
            images = images.to(self.device)
            features_real[start_idx:start_idx + n_samples] = self._get_features(images)
            
            x = torch.randn([n_samples, 3, 256, 256], requires_grad=True).to(device)
            label = torch.tensor(n_samples*(class_idx,), device=device)
            
            gen_images : torch.Tensor = sample_method(x = x, label = label)
            
            features = self._get_features(gen_images)
            features_gen[start_idx:start_idx + gen_images.shape[0]] = features
            start_idx = start_idx + images.shape[0]
            
            # save the generated features so far as npy files
            np.save(f'out/FID_data/features-batch_{i}-class_{class_name}-real.npy', features_real)
            np.save(f'out/FID_data/features-batch_{i}-class_{class_name}-gen.npy', features_real)
            
        mu_real, sigma_real = self._feature_statistics(features_real)
        mu_gen, sigma_gen = self._feature_statistics(features_gen)
        
        fid = self._frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        
        # Append the mean, std and FID to a .txt file
        with open('out/FID_data/fid.txt', 'a') as f:
            f.write(f'{mu_real} {sigma_real} {class_name} {mu_gen} {sigma_gen} {fid}\n')
        
        return fid
        
    def _get_features(self, images):
        # TODO: Add method description
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(images)
        features = features.squeeze(3).squeeze(2).cpu().numpy()
        self.feature_extractor.cpu()
        return features

    def _feature_statistics(self, features):
        # TODO: Add method description
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def _frechet_distance(self, mu1, sigma1, mu2, sigma2):

        fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*linalg.sqrtm(sigma1@sigma2))

        return fid


def setup_diffusion_model(*, device,
                          wiki_path = 'models/wikiart_256.pth',
                          head_path = 'models/head_best.pth',
                          num_classes = 27,
                          steps = 100,
                          eta = 1,
                          forward_guidance_scale = 1,
                          num_backward_steps = 0,
                          backward_guidance_scale = 1e-1):
    # TODO: Add method description

    model = WikiArt256Model().to(device)
    model.load_state_dict(torch.load(wiki_path, map_location=device))
    model.eval()

    linear_head_state_dict = torch.load(head_path)
    linear_head_model = LinearHead(num_classes * ['This is a picture of a painting'])
    linear_head_model.load_state_dict(linear_head_state_dict)

    classifier = CLIPWithHead(linear_head_model, crop_and_norm=False).to(device)

    t = torch.linspace(1, 0, steps + 1)[:-1].to(device)
    
    partial_cond_sample = partial(
        sampling.cond_sample,
        model=model,
        steps=t,
        eta=eta,
        classifier=classifier,
        num_backward_steps=num_backward_steps,
        backward_guidance_scale=backward_guidance_scale,
        forward_guidance_scale=forward_guidance_scale,
    )
    
    return partial_cond_sample


if __name__ == '__main__':
    np.random.seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = AnnotatedImageDataset(
        csv_file = 'data/annotations/wikiart_test.csv',
        root_dir = 'data/wikiart',
        device=device
    )
    classes = dataset.index_to_label
    
    feature_extractor = VGG()
    fid_calculator = FID(feature_extractor, device)
    
    batch_size = 64
    
    for idx, name in enumerate(classes):

        dataloader = DataLoader(dataset.get_dataset_label_subset(name), batch_size=batch_size, shuffle=True)
        partial_cond_sample = setup_diffusion_model(device = device)
    
        fid_calculator.calculate_fid(dataloader = dataloader,
                                     sample_method = partial_cond_sample,
                                     class_idx = idx,
                                     class_name = name)
        