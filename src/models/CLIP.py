import os
import requests
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPModel, AutoTokenizer

from src.utils.misc import load_config
from src.utils.config import Config
from src.dataloader.encodings import EncodedDataset, load_whole_dataset


class EmbType(Enum):
    CLASSIFICATION = 'classification'
    ZEROSHOT = 'zeroshot'


class ClipHead(nn.Module, ABC):
    requires_emb_type = EmbType
    index_class: list[str]

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Takes x of shape (batch_dim, emb_dim)

        Returns shape of (batch_dim, num_classes)
        """


class LinearHead(ClipHead):

    def __init__(self, classes: list[str], config_path: str = 'configs/CLIP_config.yaml') -> None:
        super().__init__()
        emb_dim = load_config(config_path)['CLIP']['latent_dim']
        self.head = nn.Linear(emb_dim, len(classes))

        self.requires_emb_type = ClipHead.requires_emb_type.ZEROSHOT

        self.index_class = classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ZeroShotHead(ClipHead):

    def __init__(self, class_prompts: list[str], config_path: str = 'configs/CLIP_config.yaml') -> None:
        super().__init__()

        model_version = load_config(config_path)['CLIP']['pretrained_ckpt']

        clipmodel = CLIPModel.from_pretrained(model_version)
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        inputs = tokenizer(class_prompts, padding=True, return_tensors="pt")
        output = clipmodel.text_model(**inputs)
        
        emb = output[1]
        proj_emb = clipmodel.text_projection(emb)
        proj_emb /= proj_emb.norm(p=2, dim=-1, keepdim=True)

        self._weights = nn.Parameter(proj_emb)
        self.temperature = nn.Parameter(clipmodel.logit_scale.exp())

        self.requires_emb_type = ClipHead.requires_emb_type.ZEROSHOT

        self.index_class = class_prompts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return F.linear(x, self._weights) * self.temperature


class PCAReducedHead(ClipHead):

    def __init__(self, classes: list[str], pca_emb_dim: Optional[int]=None, config_path: str = 'configs/CLIP_config.yaml') -> None:
        super().__init__()
        self.emb_dim = load_config(config_path)['CLIP']['latent_dim']
        self.cfg = Config('configs/config.yaml')

        self._pca_emb_dim = pca_emb_dim
        self.requires_emb_type = ClipHead.requires_emb_type.ZEROSHOT
        self.head = nn.Linear(self._pca_emb_dim or self.emb_dim, len(classes))

        self.index_class = classes
    
    def fit_pca(self, csv_name: Optional[str]=None, splits: Optional[list[str]]=None) -> 'PCAReducedHead':
        """
        Fit PCA on the whole dataset, from csv_name. Optionally filter by splits.

        If no csv_name is provided, it will use the 'all_wanted_images.csv' file.
        If this does not exist, it will create it.
        """
        if csv_name is None:
            csv_name = 'all_wanted_images.csv'
            _path = os.path.join(self.cfg.get('data', 'annotations_path'), csv_name)
            if not os.path.exists(_path):
                # This will create the csv file
                from src.image_annotation.create_csv_for_all_annotations import main as create_all_wanted_images_csv
                create_all_wanted_images_csv()

        encodings: np.ndarray = load_whole_dataset(csv_name, splits=splits)[0].detach().cpu().numpy()
        mean = encodings.mean(axis=0, keepdims=True)
        encodings -= mean
        std = encodings.std(axis=0, keepdims=True)
        encodings /= std

        self.pca = PCA().fit(encodings)
        
        self.mean = nn.Parameter(torch.from_numpy(mean), requires_grad=False).to(self.head.weight.device)
        self.std = nn.Parameter(torch.from_numpy(std), requires_grad=False).to(self.head.weight.device)
        self.components = nn.Parameter(torch.from_numpy(self.pca.components_), requires_grad=False).to(self.head.weight.device)
        return self
    
    def change_pca_emb_dim(self, pca_emb_dim: int) -> 'PCAReducedHead':
        self._pca_emb_dim = pca_emb_dim
        self.head = nn.Linear(pca_emb_dim, len(self.index_class)).to(self.head.weight.device)
        return self
    
    def embed_pca(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x /= self.std
        return F.linear(x, weight=self.components[:(self._pca_emb_dim or self.components.shape[0])])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_pca(x)
        return self.head(x)
        

class CLIPWithHead(nn.Module):

    def __init__(self, head: ClipHead, config_path: str = 'configs/CLIP_config.yaml') -> None:
        super().__init__()
        self.CFG = load_config(config_path)
        self.__base_initialization()
        if self.CFG['CLIP']['freeze']:
            self.freeze_base()
        self.head = head
        
    def __base_initialization(self):

        clipmodel = CLIPModel.from_pretrained(self.CFG['CLIP']['pretrained_ckpt'])
        self.processor = CLIPImageProcessor.from_pretrained(self.CFG['CLIP']['pretrained_ckpt'])

        self.vision_model = clipmodel.vision_model
        self.visual_projection = clipmodel.visual_projection

    def freeze_base(self):
        for param in self.parameters():
            param.requires_grad = False

    def proces_input(self, image):
        if hasattr(image, 'device'):
            device = image.device
        else:
            device = torch.device('cpu')
        return self.processor(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)

    def embed_image(self, image: torch.Tensor, preprocess=True):
        """
        Either embed for classification (few-show) or for zero-shot
        """
        if preprocess:
            image = self.proces_input(image)
        clip_outputs = self.vision_model(image)

        match self.head.requires_emb_type:
            case ClipHead.requires_emb_type.CLASSIFICATION:
                sequence_output = clip_outputs.last_hidden_state
                return torch.mean(sequence_output[:, 1:, :], dim=1)
            case ClipHead.requires_emb_type.ZEROSHOT:
                emb = clip_outputs.pooler_output
                proj_emb = self.visual_projection(emb)
                proj_emb /= proj_emb.norm(p=2, dim=-1, keepdim=True)
                return proj_emb
            case _:
                raise NotImplementedError

    def forward(self, x, preprocess=True):
        x = self.embed_image(x, preprocess=preprocess)
        logits = self.head(x)
        return logits
    
    def predict_class(self, x, preprocess=True) -> list[str]:
        logits = self.forward(x, preprocess=preprocess)
        return [self.head.index_class[i] for i in logits.argmax(dim=1)]
    
    @staticmethod
    def get_example_image(url: Optional[str]=None):
        url = url or "http://images.cocodataset.org/val2017/000000039769.jpg"
        return Image.open(requests.get(url, stream=True).raw)

    @classmethod
    def with_classification_head(cls, classes: list[str], config_path: str = 'configs/CLIP_config.yaml'):
        return cls(LinearHead(classes, config_path))
    
    @classmethod
    def with_zeroshot_head(cls, class_prompts: list[str], config_path: str = 'configs/CLIP_config.yaml'):
        return cls(ZeroShotHead(class_prompts, config_path))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = CLIPWithHead.get_example_image()
    class_prompts = [f"An image of a {obj}" for obj in ['cat', 'dog', 'car']]

    model = PCAReducedHead(class_prompts).fit_pca(splits=['train'])
    print(model(torch.randn(1, 512)))

    model = CLIPWithHead.with_classification_head(class_prompts)
    print('Classification emb shape', model.embed_image(img).shape)

    model = CLIPWithHead.with_zeroshot_head(class_prompts).eval()
    predicted_prompt = model.predict_class(img)
    plt.imshow(img)
    plt.title(predicted_prompt)
    plt.show()
