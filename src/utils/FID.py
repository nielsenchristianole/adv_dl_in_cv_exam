import os
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

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
    
    def __init__(self, feature_extractor, device):
        self.device = device
        
        self.feature_extractor = feature_extractor
        
    def calculate_fid(self, dataloader, diffusion_model, class_name):

        os.makedirs('out/FID_data', exist_ok=True)
        
        features_real = np.empty((len(dataloader.dataset), 256))
        features_gen = np.empty((len(dataloader.dataset), 256))
        
        start_idx = 0
        for i, (images, _) in tqdm(enumerate(dataloader), leave=False):
            images = images.to(self.device)
            features_real[start_idx:start_idx + images.shape[0]] = self._get_features(images)
            
            # TODO: Sample from the diffusion model. Sample images.shape[0] images for the specified class
            gen_images : torch.Tensor = None
            
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
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(images)
        features = features.squeeze(3).squeeze(2).cpu().numpy()
        self.feature_extractor.cpu()
        return features

    def _feature_statistics(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def _frechet_distance(self, mu1, sigma1, mu2, sigma2):

        fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*linalg.sqrtm(sigma1@sigma2))

        return fid

if __name__ == '__main__':
    np.random.seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classes = ["Abstract_Expressionism","Minimalism","Action_painting","Naive_Art Primitivism",
               "Analytical_cubism","New_Realism","Art_Nouveau_Modern","Northern_Renaissance",
               "Baroque","Pointillism","Color_Field_Painting","Pop_Art","Contemporary_Realism",
               "Post_Impressionism","Cubism","Realism","Early_Renaissance","Rococo",
               "Expressionism","Romanticism","Fauvism","Symbolism","High_Renaissance","Synthetic_Cubism",
               "Impressionism","Ukiyo-e","Mannerism_Late_Renaissance"]
    
    feature_extractor = VGG()
    fid_calculator = FID(feature_extractor, device)
    
    diffusion_model = None # TODO: Define the diffusion model
    
    for cls in tqdm(classes):
    
        dataloader = None # TODO: define dataloader that takes in cls
    
        fid_calculator.calculate_fid(dataloader, diffusion_model, cls)