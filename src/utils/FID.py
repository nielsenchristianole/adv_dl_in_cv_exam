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
from src.dataloader.cropped_dataloader import GeneratedData

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features[:10]
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
        
    def extract_features(self, *, dataloader, class_idx, class_name):
        
        os.makedirs('out/FID_data', exist_ok=True)
        
        # TODO: Adjust the number of samples by samples_per_fid
        features_class = None
        
        start_idx = 0
        for i, (images, _) in tqdm(enumerate(dataloader), leave=False):
            n_samples = images.shape[0]
            
            # Plot the first 10 images in the batch
            if start_idx + n_samples <= self.samples_per_fid:
                fig, axes = plt.subplots(4, 10, figsize=(20, 8), dpi=300)
                for k in range(n_samples):
                    ax = axes[k // 10, k % 10]
                    ax.imshow(images[k].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f'out/FID_data/images_{class_name}_{start_idx}.pdf', format='pdf', bbox_inches='tight', dpi=300, transparent=True)
                plt.title(f'Sampled images')
                plt.close(fig)
                exit()
            
            images = images.to(self.device)
            new_features = self._get_features(images)
            if features_class is None:
                features_class = new_features
            else:
                features_class = np.concatenate((features_class, new_features), axis=0)
            
            # save the generated features so far as npy files
        np.save(f'out/FID_data/features-class_{class_name}.npy', features_class)
        
    def calculate_fid(self, path1, path2):
        features1 = np.load(path1)
        features2 = np.load(path2)
        
        mu1, sigma1 = self._feature_statistics(features1)
        mu2, sigma2 = self._feature_statistics(features2)
        
        fid = self._frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return fid
        
    def pairwise_calculate_fid(self, *, classes):
        # TODO: Add method description

        os.makedirs('out/FID_data', exist_ok=True)
        
        pbar = tqdm(total = len(classes)**2)
        
        fids = np.zeros((len(classes), len(classes)))
        # pairwise calculate the FID between classes
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                pbar.update(1)
                if class1 == class2:
                    fids[i, j] = 0
                    with open('out/FID_data/fid.txt', 'a') as f:
                        f.write(f'{fids[i, j]}\n')
                    continue
                
                features1 = np.load(f'out/FID_data/features-class_{class1}.npy')
                features2 = np.load(f'out/FID_data/features-class_{class2}.npy')
                
                mu1, sigma1 = self._feature_statistics(features1)
                mu2, sigma2 = self._feature_statistics(features2)
                
                fid = self._frechet_distance(mu1, sigma1, mu2, sigma2)
                fids[i, j] = fid
                
                # Append the mean, std and FID to a .txt file
                with open('out/FID_data/fid.txt', 'a') as f:
                    f.write(f'{fid}\n')
        # plot the FID matrix
        plt.imshow(fids, cmap='viridis')
        plt.colorbar()
        # plt.xticks(np.arange(len(classes)), classes, rotation=45)
        # plt.yticks(np.arange(len(classes)), classes, rotation=45)
        plt.xlabel('Class index')
        plt.ylabel('Class index')
        plt.title('FID matrix')
        plt.savefig('out/FID_data/fid_matrix.pdf', format='pdf', bbox_inches='tight', dpi=300, transparent=True)
        plt.show()

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
        # add a small epsilon to the covariance matrix to avoid singular matrix
        
        sigma = np.cov(features, rowvar=False)
        
        if np.isnan(sigma).any():
            # replace nan values with small epsilon
            sigma = np.nan_to_num(sigma)
            sigma = sigma + 1e-10
        return mu, sigma

    def _frechet_distance(self, mu1, sigma1, mu2, sigma2):


        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*covmean)

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
    
    batch_size = 40
    
    # dataloader = DataLoader(dataset.get_dataset_split_subset('test'), batch_size=batch_size, shuffle=True, drop_last=False)
    dataset = GeneratedData(Path('data/sampled_images'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    fid_calculator.extract_features(dataloader = dataloader,
                                     class_idx = 0,
                                     class_name = 'sampled_images')
    
    # fid = fid_calculator.calculate_fid('out/FID_data/features-class_wikiart_test.npy', 'out/FID_data/features-class_sampled_images.npy')
    # print(fid)
    
    # for idx, name in tqdm(enumerate(classes), total = len(classes)):

    #     dataloader = DataLoader(dataset.get_dataset_label_subset(name), batch_size=batch_size, shuffle=True, drop_last=False)
    #     # partial_cond_sample = setup_diffusion_model(device = device)
    
    #     fid_calculator.extract_features(dataloader = dataloader,
    #                         #  sample_method = partial_cond_sample,
    #                          class_idx = idx,
    #                          class_name = name)
        
    # fid_calculator.pairwise_calculate_fid(classes = classes)
    
        # fid_calculator.calculate_fid(dataloader = dataloader,
        #                              sample_method = partial_cond_sample,
        #                              class_idx = idx,
        #                              class_name = name)
        