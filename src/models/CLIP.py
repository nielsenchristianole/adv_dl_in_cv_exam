import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPTextConfig, CLIPVisionConfig

import pdb

class CLIPWithHead:
    def __init__(self) -> None:
        pass
        # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
        configuration = CLIPConfig()
        # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
        model = CLIPModel(configuration)
        # Accessing the model configuration
        configuration = model.config
        # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
        # Initializing a CLIPText and CLIPVision configuration
        config_text = CLIPTextConfig()
        config_vision = CLIPVisionConfig()
        config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
        pdb.set_trace()

if __name__ == '__main__':
    CWH = CLIPWithHead()

