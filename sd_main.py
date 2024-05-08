import torch

from src.models.CLIP import CLIPWithHead, LinearHead
from src.models.stable_diffusion import StableDiffusion


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 27
linear_head_model = LinearHead(num_classes * [None])
linear_head_model.load_state_dict(torch.load('./models/head_best.pth', map_location=device))

classifier = CLIPWithHead(linear_head_model, crop_and_norm=False).to(device).eval()


if __name__ == '__main__':
    num_samples = 2
    cg_label = torch.tensor(num_samples * (26,), device=device)

    SD = StableDiffusion(cahce_dir='.', cg_model=classifier, device=device)
    _sample = SD.sample(
        'A painting',
        num_samples=num_samples,
        cg_label = cg_label,
        cfg_guidance_scale = 1,
        forward_guidance_scale = 500,
        backward_guidance_scale = 1,
        num_backward_steps = 5,
        backward_step_size = 0.1,
        num_inference_steps=50
    )


    from src.models.stable_diffusion import latents_to_pil, latents_to_image

    with torch.no_grad():
        out = classifier(latents_to_image(_sample, SD.vae))
        loss = torch.nn.functional.cross_entropy(out, cg_label)
        probs = torch.nn.functional.softmax(out, dim=1)
        print(f"Loss: {loss.item()}")
        print(f"Predicted class: {probs.argmax(dim=1).detach().cpu().numpy()}")
        print(f"Target class probability: {probs[:, cg_label[0]].detach().cpu().numpy()}")
        print(f"Predicted probability: {probs.max(dim=1).values.detach().cpu().numpy()}")

    for i, img in enumerate(latents_to_pil(_sample, SD.vae), start=1):
        img.save(f'SD_sample{i}.png')
