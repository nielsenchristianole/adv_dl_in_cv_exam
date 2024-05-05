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
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class StableDiffusion:
    def __init__(self, cahce_dir: str) -> None:
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'INITIALIZING DIFFUSION ON DEIVCE: {self.device}')
        self.model_name = "stabilityai/stable-diffusion-2-1-base"
        self.dtype = torch.float16
        self.cache_dir = cahce_dir
        
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['HF_HUB_CACHE'] = self.cache_dir
        
        #### INIT UNET
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet", torch_dtype=self.dtype).to(self.device)
        
        #### INIT VAE
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae", torch_dtype=self.dtype, cache_dir=self.cache_dir).to(self.device)
        self.imageprocessor = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor)
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
    
    def latent2image(self, latent):
        image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)
        image = (self.denormalize(image) * 255).round().to(torch.int16)
        return 
    
    def image2latent(self, image):
        self.vae.encode()

    def sample(self, prompt, guidance_scale, n_steps: int, init_latents, do_cfg: bool = True):
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
        with torch.no_grad():
            for i, ts in enumerate(tqdm(self.scheduler.timesteps)):
                
                unet_ipt = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)
                # print(f'Latents shape: {latents.shape}')
                
                noise_pred_uncond, noise_pred_text = self.unet(unet_ipt, ts, encoder_hidden_states=emb).sample.chunk(2)
                
                # do cfg
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # run the guidance transform
                # pred = gtfm.encode(u, t, idx=i)

                # update the latents
                latents = self.scheduler.step(noise_pred, ts, latents).prev_sample
            
            # decode and return the final latents
            # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            
        # pdb.set_trace()
        # print(f'final img shape: {image.shape}')
        # processed_img = self.imageprocessor.postprocess(image, do_denormalize=[True]*image.size(0))
        PIL_image = latents_to_pil(latents, self.vae)
        return PIL_image


    def universal_guidance(self):
        pass


if __name__ == '__main__':
    from torchvision.utils import save_image

    # SD = StableDiffusion()
    # sample = SD.sample("A painting", 1, 10, None)
    pdb.set_trace()