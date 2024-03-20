import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel
import requests
from PIL import Image
from src.utils.misc import load_config
import pdb

class CLIPWithHead(nn.Module):
    def __init__(self, config_path: str = 'configs/CLIP_config.yaml') -> None:
        super().__init__()
        self.CFG = load_config(config_path)
        self.__base_initialization()
        if self.CFG['CLIP']['freeze']:
            self.freeze_base()
        self.__head_initialization()
        self.softmax = nn.Softmax(dim=-1)
        
    def __base_initialization(self):
        self.processor = CLIPImageProcessor.from_pretrained(self.CFG['CLIP']['pretrained_ckpt'])
        self.base = CLIPVisionModel.from_pretrained(self.CFG['CLIP']['pretrained_ckpt'])

    def __head_initialization(self):
        self.head = nn.Sequential(nn.ReLU(),
                                  nn.Linear(self.CFG['CLIP']['latent_dim'], self.CFG['Head']['out_dim']))
    
    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def proces_input(self, image):
        return self.processor(images=image, return_tensors="pt", padding=True)

    def forward(self, image):
        clip_outputs = self.base(**self.proces_input(image)) # keys: ['last_hidden_state', 'pooler_outputs']
        logits = self.head(clip_outputs.pooler_output)
        return {'logits': logits, 'probabilities': self.softmax(logits)}
        
    def example(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return self.forward(image=image)
