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
    get_top_n = 5
    loop_n_times = 1000
    # target_cls = 12
    import os
    
    SD = StableDiffusion(cahce_dir='.', cg_model=classifier, device=device)
    
    from src.models.stable_diffusion import latents_to_pil, latents_to_image
    # import tqdm
    # import numpy as np
    
    # cfg_guidance_scale      = [1,1,1,1]
    # forward_guidance_scale  = [500,500,500,500]
    # backward_guidance_scale = [50,50,250,500]
    # num_backward_steps      = [10,10,10,10]
    # backward_step_size      = [0.1,0.1,0.1,0.1]
    # num_inference_steps     = [30,30,30,30]
    
    
    # for a, b, c, d, e, f in zip(cfg_guidance_scale, forward_guidance_scale, backward_guidance_scale, num_backward_steps, backward_step_size, num_inference_steps):
    # print(f"cfg: cfg_guide_scale={a}, fw_guide_scale={b}, bw_guide_scale={c}, n_bw_steps={d}, bw_step_size={e}, n_inference_steps={f}")
    
    # pred_classes = []
    # in_top = []
    for loop in range(loop_n_times):
        print("Running loop:",loop,"out of", loop_n_times)
        for target_cls in range(num_classes + 1):
            # target_cls = 2
            # target_cls = 19
            # target_cls = int(np.random.randint(0, num_classes, (1,)))
            
            if target_cls == num_classes:
                target_cls == "non_conditioned"
                cg_label = torch.tensor(num_samples * (0,), device=device)
                
                _sample = SD.sample(
                    'painting',
                    num_samples=num_samples,
                    cg_label                = cg_label,
                    cfg_guidance_scale      = 0.5,#a,
                    forward_guidance_scale  = 0,#b,
                    backward_guidance_scale = 0,#c,
                    num_backward_steps      = 0,#d,
                    backward_step_size      = 0,#e,
                    num_inference_steps     = 20#f
                )
            else:
                cg_label = torch.tensor(num_samples * (target_cls,), device=device)
                
                _sample = SD.sample(
                    'painting',
                    num_samples=num_samples,
                    cg_label                = cg_label,
                    cfg_guidance_scale      = 0.5,#a,
                    forward_guidance_scale  = 10,#b,
                    backward_guidance_scale = 5,#c,
                    num_backward_steps      = 30,#d,
                    backward_step_size      = 0.05,#e,
                    num_inference_steps     = 20#f
                )

            with torch.no_grad():
                out = classifier(latents_to_image(_sample, SD.vae))
                loss = torch.nn.functional.cross_entropy(out, cg_label)
                probs = torch.nn.functional.softmax(out, dim=1)
                
                last_preds = probs.argsort(dim=1).detach().cpu().numpy()[:,-get_top_n:]
                last_pred = probs.argmax(dim=1).detach().cpu().numpy()
                # pred_classes.append(list(last_pred))
                # in_top.append([int(target_cls) in k for k in last_preds])
                
                # print(f"Loss: {loss.item()}<<<")
                # print(f"Predicted class: {probs.argsort(dim=1)[:,-5:].detach().cpu().numpy()}")
                # print(f"Target class probability: {probs[:, cg_label[0]].detach().cpu().numpy()}")
                # print(f"Predicted probability: {probs.max(dim=1).values.detach().cpu().numpy()}")
            
            # for j, img in enumerate(latents_to_pil(_sample, SD.vae), start=1):
            #     img.save(f'sample_cls{target_cls}_{loop}-{j}.png')
            
            if os.path.exists(f'/zhome/e7/4/155304/blackhole/{target_cls}') == False:
                os.makedirs(f'/zhome/e7/4/155304/blackhole/{target_cls}')
                
            for j, img in enumerate(latents_to_pil(_sample, SD.vae), start=1):
                img.save(f'/zhome/e7/4/155304/blackhole/{target_cls}/sample4_{loop}-{j}.png')

    # # print("predicted_classes",pred_classes)
    # true_preds = sum([sum(i) for i in in_top])
    # n_preds = loop_n_times*num_samples
    # top_n_acc.append(true_preds/n_preds)
    # print(f"Is in top {get_top_n} preds: {true_preds}/{n_preds}={true_preds/n_preds:.2}")
    
            
    # print("top accs:",top_n_acc)
    
    
