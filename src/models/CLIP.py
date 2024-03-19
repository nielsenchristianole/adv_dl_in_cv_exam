import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel
import requests
from PIL import Image
from src.utils.misc import load_config
import pdb

class CLIPWithHead:
    def __init__(self, config_path: str = 'configs/CLIP_config.yaml') -> None:
        self.CFG = load_config(config_path)
        self.__base_initialization()
        
    def __base_initialization(self):
        self.processor = CLIPImageProcessor.from_pretrained(self.CFG['CLIP']['pretrained_ckpt'])
        self.base = CLIPVisionModel.from_pretrained(self.CFG['CLIP']['pretrained_ckpt'])
    
    def proces_input(self, image):
        return self.processor(images=image, return_tensors="pt", padding=True)

    def forward(self, image):
        clip_outputs = self.base(**self.proces_input(image)) # keys: ['last_hidden_state', 'pooler_outputs']

        
    def example(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        self.forward(image=image)

if __name__ == '__main__':
    CWH = CLIPWithHead('test')
    CWH.example()


