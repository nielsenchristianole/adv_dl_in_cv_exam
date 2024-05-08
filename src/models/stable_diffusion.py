from typing import Callable, Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler, DDIMScheduler, KarrasVeScheduler, PNDMScheduler, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import pdb
import os
from PIL import Image

def text_embeddings(tokenizer, text_encoder, prompts, device, maxlen=None):
    "Extracts text embeddings from the given `prompts`."
    maxlen = maxlen or tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    
    input_ids = inp.input_ids.to(device)
    attn_mask = inp.attention_mask.to(device)

    return text_encoder(inp.input_ids.to(device), attention_mask=attn_mask)[0]

def latents_to_pil(latents, vae):
    '''
    Function to convert latents to images
    '''
    latents = (1 / vae.config.scaling_factor) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def latents_to_image(latents, vae):
    latents = (1 / vae.config.scaling_factor) * latents # scale
    image = vae.decode(latents).sample # decode
    # image = (image / 2 + 0.5).clamp(0, 1) # denormalize
    # image = image.permute(0, 2, 3, 1) # reshape
    # image = (image * 255).round().to(torch.int16) # [0, 255]
    return image

def image_to_latent(image, vae):
    pass


class StableDiffusion:
    def __init__(self, cahce_dir: str, cg_model: Callable[[Any], torch.Tensor], device=None) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'INITIALIZING DIFFUSION ON DEIVCE: {self.device}')
        self.model_name = "stabilityai/stable-diffusion-2-1-base"
        self.dtype = torch.float16
        self.cache_dir = cahce_dir
        self.cg_model = cg_model
        
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['HF_HUB_CACHE'] = self.cache_dir
        
        #### INIT UNET
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet", torch_dtype=self.dtype).to(self.device)
        
        #### INIT VAE
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae", torch_dtype=self.dtype, cache_dir=self.cache_dir).to(self.device)
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        
        #### INIT CLIP (for embedding prompts)
        self.tokenizer = CLIPTokenizer.from_pretrained(
          self.model_name,
          subfolder="tokenizer",
          torch_dtype=self.dtype,
          cahce_dir=self.cache_dir)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder",
            torch_dtype=self.dtype, cache_dir=self.cache_dir).to(self.device)
    
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    def denormalize(self, images):
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def sample(self, 
               prompt: str, 
               guidance_scale: int, 
               n_steps: int, 
               init_latents: torch.tensor=None,
               num_backward_steps: int = 4,
               backward_step_size: float = 0.1,
               backward_guidance_scale: float = 5,
               cg_label: torch.tensor = torch.tensor([26]), dtype=torch.int16):
        cg_label = cg_label.to(self.device)

        if init_latents is None:
            init_latents = torch.randn((1, self.unet.config.in_channels, 64, 64), device=self.device, dtype=self.unet.dtype)
        
        self.scheduler.set_timesteps(n_steps)

        # prepare text embeddings
        text = text_embeddings(self.tokenizer, self.text_encoder, prompt, self.device, maxlen=None)
        uncond = text_embeddings(self.tokenizer, self.text_encoder, '', self.device, maxlen=None)
        emb = torch.cat([uncond, text]).type(self.unet.dtype)

        # prepare latents
        latents = torch.clone(init_latents)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # run diffusion
        for i, ts in enumerate(tqdm(self.scheduler.timesteps)):
           
            unet_ipt = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)
            
            noise_pred_uncond, noise_pred_text = self.unet(unet_ipt, ts, encoder_hidden_states=emb).sample.chunk(2)
            
            # do classifier free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # update the latents
            inside_stepper = deepcopy(self.scheduler)
            scheduler_out = self.scheduler.step(noise_pred, ts, latents)
            # latens = scheduler_out.prev_sample
            latens_0 = scheduler_out.pred_original_sample

            # do classifier guidance
            if i != n_steps-1:
                with torch.enable_grad():
                    _latens_0 = latens_0.clone().detach().requires_grad_(True)
                    delta = torch.zeros_like(_latens_0)

                    # backward universal guidance
                    for _ in range(num_backward_steps):
                        delta = delta.detach().clone().requires_grad_(True)
                        cg_ipt = latents_to_image(_latens_0 + delta, self.vae)

                        # logits = self.cg_model(cg_ipt.to('cpu'))
                        logits = self.cg_model(cg_ipt)
                        loss = torch.nn.functional.cross_entropy(logits, cg_label)
                        loss.backward()

                        delta = delta - backward_step_size * delta.grad
                
                # calculate x_{t-1}
                latents = inside_stepper.step(
                    noise_pred - delta * backward_guidance_scale,
                    ts, latents).prev_sample

            else:
                latents = inside_stepper.step(
                    noise_pred,
                    ts, latents).prev_sample
        
        return latents