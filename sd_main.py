from src.models.CLIP import CLIPWithHead
from src.models.stable_diffusion import StableDiffusion
import pdb
import os
import PIL

if __name__ == '__main__':
    # import torch
    # print(torch.cuda.is_available())
    SD = StableDiffusion(cahce_dir='/work3/s204138/StableDiffusion')
    sample = SD.sample("An image of a dog", 10, 50, None, do_cfg=True)
    print(f"sample len: {len(sample)}")
    sample[0].save(f'SD_sample1.png')
    # sample[1].save(f'SD_sample2.png')
    pdb.set_trace()