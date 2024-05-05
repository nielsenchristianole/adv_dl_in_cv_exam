import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler, DDIMScheduler, KarrasVeScheduler, PNDMScheduler, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from utils import text_embeddings
from tqdm.auto import tqdm
import pdb

def text_embeddings(tokenizer, text_encoder, prompts, device, maxlen=None):
    "Extracts text embeddings from the given `prompts`."
    maxlen = maxlen or tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    return text_encoder(inp.input_ids.to(device))[0]

class StableDiffusion:
    def __init__(self) -> None:
        self.device = 'cpu' #('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'INITIALIZING DIFFUSION ON DEIVCE: {self.device}')
        self.model_name = "stabilityai/stable-diffusion-2-1-base"
        self.dtype = torch.float16
        
        #### INIT UNET
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet", torch_dtype=self.dtype).to(self.device)
        
        #### INIT VAE
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        self.imageprocessor = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor)
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        
        #### INIT CLIP (for embedding prompts)
        self.tokenizer = CLIPTokenizer.from_pretrained(
          self.model_name,
          subfolder="tokenizer",
          torch_dtype=self.dtype)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder",
            torch_dtype=self.dtype).to(self.device)
    
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

    def sample(self, prompt, guidance_scale, n_steps: int, init_latents):
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
            print(f'i: {i}')
            if i == 0:
                latents = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)
            else:
                latents = self.scheduler.scale_model_input(latents, ts)
            print(f'Latents shape: {latents.shape}')
            noise_pred = self.unet(latents, ts, encoder_hidden_states=emb)[0]
            
            # run the guidance transform
            # pred = gtfm.encode(u, t, idx=i)

            
            # update the latents
            latents = self.scheduler.step(noise_pred, ts, latents).prev_sample
        
        pdb.set_trace()
        # decode and return the final latents
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)

        self.image_processor.postprocess(image, do_denormalize=True)
        return image


    def universal_guidance(self):
        pass


if __name__ == '__main__':
    from torchvision.utils import save_image

    SD = StableDiffusion()
    sample = SD.sample("A painting", 1, 10, None)
    pdb.set_trace()