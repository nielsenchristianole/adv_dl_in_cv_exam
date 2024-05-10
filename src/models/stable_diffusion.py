import os
import logging
from typing import Callable, Any, Optional

from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers.image_processor import VaeImageProcessor



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


class StableDiffusion:
    def __init__(self, cahce_dir: str, cg_model: Callable[[Any], torch.Tensor], device=None, num_timesteps: int=1000) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'INITIALIZING DIFFUSION ON DEIVCE: {self.device}')
        self.model_name = "stabilityai/stable-diffusion-2-1-base"
        self.dtype = torch.float16
        self.cache_dir = cahce_dir
        self.cg_model = cg_model
        self.num_timesteps = num_timesteps
        
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['HF_HUB_CACHE'] = self.cache_dir
        
        #### INIT UNET
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet", torch_dtype=self.dtype).to(self.device)
        
        #### INIT VAE
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae", torch_dtype=self.dtype, cache_dir=self.cache_dir).to(self.device)
        self.scheduler = DDPMScheduler()
        
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
               cfg_guidance_scale: int,
               num_samples: int=1,
               z_T: Optional[torch.tensor]=None,
               num_backward_steps: int = 4,
               backward_step_size: float = 0.1,
               forward_guidance_scale: float = 10,
               backward_guidance_scale: float = 5,
               num_inference_steps: int = 50,
               cg_label: torch.tensor = torch.tensor([26], dtype=torch.int16)):
        cg_label = cg_label.to(self.device)

        if z_T is None:
            z_T = torch.randn((num_samples, self.unet.config.in_channels, 64, 64), device=self.device, dtype=self.unet.dtype)
        
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        prev_timesteps = torch.cat((timesteps[1:], torch.tensor((-1,))))

        # prepare text embeddings
        text = text_embeddings(self.tokenizer, self.text_encoder, prompt, self.device, maxlen=None)
        uncond = text_embeddings(self.tokenizer, self.text_encoder, '', self.device, maxlen=None)
        emb = torch.cat(num_samples * [uncond] + num_samples * [text]).type(self.unet.dtype)

        # prepare latents
        z_t = torch.clone(z_T)


        # run diffusion
        for t, prev_t in zip(tqdm(timesteps, leave=False), prev_timesteps):
            #current_alpha_t = alpha_prod_t / alpha_prod_t_prev#self.scheduler.alphas[t]
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t
            
            unet_ipt = torch.cat([z_t] * 2)
            
            eps_uncond, eps_text = self.unet(unet_ipt, t, encoder_hidden_states=emb).sample.chunk(2)
            
            # do classifier free guidance
            eps = eps_uncond + cfg_guidance_scale * (eps_text - eps_uncond)

            # equation 3
            z_0 = (z_t - torch.sqrt(beta_prod_t) * eps) / torch.sqrt(alpha_prod_t)
            z_0 = z_0.clamp(-1, 1)

            # calculate reverse step
            pred_original_sample_coeff = torch.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
            current_sample_coeff = torch.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
            z_t = pred_original_sample_coeff * z_0 + current_sample_coeff * z_t

            # forward guidance
            if forward_guidance_scale:
                with torch.enable_grad():
                    _z_t = z_t.clone().detach().requires_grad_(True)
                    _z_0 = (_z_t - torch.sqrt(1 - current_alpha_t) * eps) / torch.sqrt(current_alpha_t)
                    _x_0 = latents_to_image(_z_0, self.vae)
                    logits = self.cg_model(_x_0)
                    loss = F.cross_entropy(logits, cg_label)
                    loss.backward()

                    nabla_z_t = _z_t.grad.detach()
            else:
                nabla_z_t = 0

            forward_guided_eps = eps + forward_guidance_scale * torch.sqrt(1 - current_alpha_t) * nabla_z_t
            forward_guided_z_0 = (z_t - torch.sqrt(1 - current_alpha_t) * forward_guided_eps) / torch.sqrt(current_alpha_t)

            # backward guidance
            if t != 0 and num_backward_steps:
                with torch.enable_grad():
                    _z_0 = forward_guided_z_0.clone().detach().requires_grad_(True)
                    delta = torch.zeros_like(_z_0)

                    # backward universal guidance
                    for bw_step in range(num_backward_steps):
                        delta = delta.detach().clone().requires_grad_(True)
                        cg_ipt = latents_to_image(_z_0 + delta, self.vae)

                        # logits = self.cg_model(cg_ipt.to('cpu'))
                        logits = self.cg_model(cg_ipt)
                        loss = F.cross_entropy(logits, cg_label)
                        loss.backward()

                        delta = delta - backward_step_size * delta.grad

                        if (logits.argmax(dim=1) == cg_label).all():
                            print(f"Broke out at: {bw_step} out of {num_backward_steps}")
                            break
                    
                    else:
                        logging.warning('Did not produce an adversarial')

                    
                    delta = delta.detach()
                
                backward_guided_eps = forward_guided_eps - backward_guidance_scale * torch.sqrt(current_alpha_t / (1 - current_alpha_t)) * delta
                backward_guided_z_0 = (z_t - torch.sqrt(1 - current_alpha_t) * backward_guided_eps) / torch.sqrt(current_alpha_t)

                z_t = torch.sqrt(current_alpha_t) * backward_guided_z_0 + torch.sqrt(1 - current_alpha_t) * backward_guided_eps
            
            else:
                z_t = torch.sqrt(current_alpha_t) * forward_guided_z_0 + torch.sqrt(1 - current_alpha_t) * forward_guided_eps

            # add noise
            if t != 0:
                variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
                variance = torch.clamp(variance, min=1e-20)
                z_t += torch.sqrt(variance) * torch.randn_like(z_t)

        
        return z_t