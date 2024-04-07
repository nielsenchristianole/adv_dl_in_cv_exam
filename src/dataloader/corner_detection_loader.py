
from src.utils.config import Config
from pathlib import Path
from typing import cast
import pandas as pd
import numpy as np
import ast
import cv2
import torch

from torch.utils.data import Dataset

import albumentations as A

class CornerDataset(Dataset):
    
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def __init__(self, train_split: float = 0.8, is_train: bool = True, scale_to=512):
        
        self.is_train = is_train
        self.scale_to = scale_to
        
        cfg = Config('configs/config.yaml')
        
        anno_folder = Path(cast(str,cfg.get("data", "elo_annotations_path")))
        
        self.paths, self.corners = self._get_paths(anno_folder, train_split)
            
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        
        path = self.paths[idx]
        corners = self.corners[idx]
        
        img, corners = self._extract_data(path, corners)
        
        img, corners = self.augment(img, corners)
        
        img = self.to0to1(img)
        img = self.standardize(img)
        
        img = torch.tensor(img).permute(2, 0, 1).float()
        corners = torch.tensor(corners).float()
        
        corners = self.normalize_corners(corners).flatten()
        
        return img, corners
    
    def augment(self, img, corners):
        
        transform =  A.Compose([
            A.RandomBrightnessContrast(p=0.1),
            A.OneOf([
                A.MotionBlur(p=0.1),
                A.GaussianBlur(p=0.1),
                A.GaussNoise(p=0.1),
            ], p=0.5),
            A.RandomGamma(p=0.1),
            A.RandomFog(p=0.1, fog_coef_lower=0.1, fog_coef_upper=0.5),
            A.RandomRain(p=0.1),
            A.RandomSnow(p=0.1),
            A.RandomSunFlare(p=0.05, src_radius=100),
            A.RandomShadow(p=0.1),
            A.RandomToneCurve(p=0.1),
            A.RandomBrightnessContrast(p=0.1)])
            # A.Affine(scale=(0.8, 1.2), rotate=(-10, 10), shear=(-10, 10), p=0.5)
        # ]), keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))
        
        img = transform(image=img)['image']
        
        return img, corners
        
    
    def normalize_corners(self, corners):
        return corners / self.scale_to
    
    def standardize(self, img):
        return (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
    
    def unstandardize(self, img):
        return img * self.IMAGENET_STD + self.IMAGENET_MEAN
        
    def to0to1(self, img):
        return img / 255.0
    
    def to0to255(self, img):
        return img * 255.0
    
    def _extract_data(self, path, corners):
        
        img = cv2.imread("data/" + path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        w, h = img.shape[1], img.shape[0]
        
        img = cv2.resize(img, (self.scale_to, self.scale_to))
        
        img = cast(np.ndarray, img)
        
        # scale corners from original image size to scaled image size
        corners[:, 0] = corners[:, 0] / w * self.scale_to
        corners[:, 1] = corners[:, 1] / h * self.scale_to
        
        return img, corners
        
    def _get_paths(self, anno_folder, train_split):
        
        annotations = pd.read_csv(anno_folder / "corners.csv")
        
        all_paths = annotations['path']
        corners_raw = annotations['corners']
        
        n_datapoints = len(all_paths)
        
        corners = []
        for corner in corners_raw:
            corners.append(ast.literal_eval(corner))
            
        corners = np.array(corners)
        
        seed = 42
        np.random.seed(seed)
        train_idx = np.random.choice(n_datapoints, int(n_datapoints * train_split), replace=False)
        
        train_paths = [list(all_paths)[i] for i in train_idx]
        
        
        train_labels = corners[train_idx]
        
        
        if self.is_train:
            return train_paths, train_labels
        else:
            test_idx = np.array(list(set(range(n_datapoints)) - set(train_idx)))
            test_paths = [list(all_paths)[i] for i in test_idx]
            test_labels = corners[test_idx]
            
            return test_paths, test_labels
            

if __name__ == "__main__":
    
    dataset = CornerDataset()
    
    print(len(dataset))
    
    import matplotlib.pyplot as plt
    for i in range(10):
        img, corners = dataset[i]
        
        img = np.array(img.permute(1, 2, 0))
        
        corners = corners * 512
        
        corners = corners.reshape(-1, 2)
        
        img = dataset.unstandardize(img)
        
        plt.imshow(img)
        plt.scatter(corners[:, 0], corners[:, 1])
        plt.show()
        plt.title(f"test img {i} out of 10")
        
    print("done")