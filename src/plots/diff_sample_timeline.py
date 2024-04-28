import torch

from src.models.diffusion import WikiArt256Model
from src.models.diffusion_sampling import sample, cond_sample


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Sample from a diffusion model.')
    # either choose sample or plot
    parser.add_argument('--sample', action='store_true', help='Sample from the model.')
    parser.add_argument('--plot', action='store_true', help='Plot the model predictions.')

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    steps = 50
    t = torch.linspace(1, 0, steps + 1)[:-1].to(device)

    num_samples = 4


    if args.sample:

        # load model file and run inference

        path = 'models/wikiart_256.pth'
        model = WikiArt256Model().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        eta=1
        target_label = 26

        forward_guidance_scale=100
        num_backward_steps=0
        backward_step_size=1e-1
        backward_guidance_scale=1e-1

        # create a dummy input
        x = torch.randn([1, 3, 256, 256], requires_grad=True).to(device)
        
        label = torch.tensor(num_samples*(target_label,), device=device)
        for n in range(num_samples):
            sample(
                model=model,
                x=x,
                steps=t,
                eta=eta,
                plot_preds=True,
                plot_preds_folder=f'figures/sample_preds{n}',
                extra_args={}
            )

    if args.plot:
        import os
        import matplotlib.pyplot as plt 
        # timeline folder

        fig, axs = plt.subplots(num_samples, int(steps/5), figsize=(10, 5))

        for n in range(num_samples):
            folder = f'figures/sample_preds{n}'

            # get all images in the folder
            files = os.listdir(folder)
            files = [f for f in files if f.endswith('.png')]

            # filenames are numbers, sort them
            files = sorted(files, key=lambda x: int(x.split('.')[0]))

            for i, file in enumerate(files):
                img_path = os.path.join(folder, file)
                img = plt.imread(img_path)
                ax = axs[n, i]
                ax.imshow(img)
                ax.axis("off")
                title = t[int(i*5)].item()
                # round
                title = round(title, 2)
                title = "t = " + str(title)

                ax.set_title(title, fontsize=8)
        
        # Adjust subplot parameters to reduce space
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.2)


        # plot withouy white background
        fig.patch.set_alpha(0)
        fig.patch.set_facecolor('none')
        fig.patch.set_edgecolor('none')

        # savefig
        plt.tight_layout()
        plt.savefig("figures/sample_timeline.pdf")
        # plt.show()