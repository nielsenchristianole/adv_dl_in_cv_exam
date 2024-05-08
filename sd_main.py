import pdb

import torch

from src.models.CLIP import CLIPWithHead, LinearHead
from src.models.stable_diffusion import StableDiffusion


# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 27
linear_head_model = LinearHead(num_classes * [None])
linear_head_model.load_state_dict(torch.load('./models/head_best.pth', map_location=device))

classifier = CLIPWithHead(linear_head_model, crop_and_norm=False).to(device).eval()


if __name__ == '__main__':
    cg_label = torch.tensor((26,), device=device)

    SD = StableDiffusion(cahce_dir='.', cg_model=classifier)#, device=device)
    _sample = SD.sample(
        'A closeup image of a painting',
        guidance_scale = 10,
        n_steps = 50,
        num_backward_steps=2,
        backward_guidance_scale=0.5,
        cg_label = cg_label
    )

    from src.models.stable_diffusion import latents_to_pil, latents_to_image

    with torch.no_grad():
        out = classifier(latents_to_image(_sample, SD.vae))
        loss = torch.nn.functional.cross_entropy(out, cg_label)
        probs = torch.nn.functional.softmax(out, dim=1)
        print(f"Loss: {loss.item()}")
        print(f"Predicted class: {probs.argmax(dim=1).detach().cpu().numpy()}")
        print(f"Target class probability: {probs[:, cg_label.item()].detach().cpu().numpy()}")
        print(f"Predicted probability: {probs.max(dim=1).values.detach().cpu().numpy()}")
        # print(np.array2string(probs.detach().cpu().numpy(), precision=3))

    
    
    for i, img in enumerate(latents_to_pil(_sample, SD.vae), start=1):
        img.save(f'SD_sample{i}.png')
    # pdb.set_trace()