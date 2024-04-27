# %% 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property


from src.utils.config import Config
from src.plots.dtu_colors import DTUColors

cfg = Config("/Users/alf/Repos/adv_dl_in_cv_exam/configs/config.yaml")


class WikiArtDatasetPlotter():
    def __init__(self, wikiart_path: str, colorClass: DTUColors) -> None:
        self.colors = colorClass
        self.wikiart_path = wikiart_path

        self.classes = os.listdir(wikiart_path)
        self.classes = [c for c in self.classes if os.path.isdir(os.path.join(wikiart_path, c)) and not c.startswith(".")]

        print("WikiArtDatasetPlotter initialized.")
        print("wikiart_path: ", wikiart_path)


    def make_wikiart_hist(self, save_name: str):
        wikiart_path = self.wikiart_path
        print("wikiart_path: ", wikiart_path)

        counts = self.class_counts

        # Make histogram of image counts
        fig = plt.figure(figsize=(8, 5))
        ckeys = list(counts.keys())
        # replace _ with space for better readability
        ckeys = [c.replace("_", " ") for c in ckeys]
        plt.bar(ckeys, counts.values(), color=self.colors.get_secondary_color("dtulightgreen"))
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.xlabel("Genre")
        plt.ylabel("Number of Images")
        plt.title("Number of Images per Genre (WikiArt Dataset)")
        plt.tight_layout()
        fig = self.make_background_transparent(fig)

        self.save_plot(save_name)


    def plot_grid_images(self, save_path: str):

        # class
        number_of_classes = len(self.classes)
        # plot 1 image from each class in a grid with 3 rows. There are 27 classes, so 10 classes per row
        fig, axs = plt.subplots(3, 10, figsize=(20, 6))
        for i, folder in enumerate(self.classes):
            files = os.listdir(os.path.join(self.wikiart_path, folder))
            img_path = os.path.join(self.wikiart_path, folder, files[0])
            img = plt.imread(img_path)
            ax = axs[i//10, i%10]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(folder.replace("_", " "), fontsize=8)

        # plot some random images in the last three columns
        for i in range(1, 4):
            random_class = np.random.choice(self.classes)
            files = os.listdir(os.path.join(self.wikiart_path, random_class))
            img_path = os.path.join(self.wikiart_path, random_class, files[0])
            img = plt.imread(img_path)
            ax = axs[2, 10-i]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(random_class.replace("_", " "), fontsize=8)

        # make plot without white background, transparent background
        fig = self.make_background_transparent(fig)
        
        plt.tight_layout()
        self.save_plot(save_path)

        # plot

    @cached_property
    def class_counts(self):
        counts = {}
        for folder in self.classes:
            files = os.listdir(os.path.join(self.wikiart_path, folder))
            counts[folder] = len(files)
        return counts

    def save_plot(self, save_path: str):

        # make figures folder if it doesn't exist
        if not os.path.exists("figures"):
            os.makedirs("figures")
        # save as pdf 
        plt.savefig("./figures/" + save_path + ".pdf", bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    def make_background_transparent(self, fig):
        fig.patch.set_alpha(0)
        fig.patch.set_facecolor('none')
        fig.patch.set_edgecolor('none')
        return fig

if __name__ == "__main__":
    colorClass = DTUColors()
    wikiart_path = cfg.get("data", "wikiart_path")
    plotter = WikiArtDatasetPlotter(wikiart_path, colorClass)
    plotter.make_wikiart_hist("wikiart_hist")

    plotter.plot_grid_images("wikiart_grid")


    # sum all values in class_counts
    total_images = sum(plotter.class_counts.values())
    print(f"Total number of images in WikiArt dataset: {total_images}")
    print("Number of classes: ", len(plotter.classes))
    pass