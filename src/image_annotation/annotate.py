import os
from os import PathLike
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tkinter as tk
from functools import cached_property
from glob import glob
from collections import deque

from src.utils.config import Config
from src.utils.misc import split_path



class DatasetAnnotation:
    def __init__(self, data_folder: PathLike, output_csv: PathLike, *, key_to_label_map: dict[str, str] | None=None, ignore_patters: list[str] | None=None):
        
        self.data_folder = data_folder
        self.output_csv = output_csv
        self.columns = ['image', 'path', 'label']
        self.ignore_patters = ignore_patters or ['**/never/**']

        self.key_to_label_map = key_to_label_map or {i: i for i in ['1', '2', '3', '4']}

        self.keys = set(self.key_to_label_map.keys())
        self.key_err_msg = f"Invalid key pressed, please press {str(sorted(self.keys))[1:-1]}"
        
        # Make csv file if it does not exist
        if not os.path.exists(self.output_csv):
            # make dir
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.output_csv, index=False)
        
        df = pd.read_csv(self.output_csv)
        self.annotated_set = set(df['path'])
        self.not_annotated_queue = deque(np.random.permutation(list(set(self.all_paths) - self.annotated_set)))

    def append_annotation_to_csv(self, path: PathLike, label: str):
        """Append the annotation to the CSV file."""
        # Append the annotation to the CSV file'
        _, image_id, _ = split_path(path)
        row = [image_id, path, label]
        df = pd.DataFrame([row], columns=self.columns)
        df.to_csv(self.output_csv, mode='a', index=False, header=not os.path.exists(self.output_csv))
        self.annotated_set.add(path)
        self.not_annotated_queue.popleft()

    @cached_property
    def all_paths(self) -> set[str]:
        """Return all the image paths in the data folder."""
        _all = set(glob('**/*.jpg', root_dir=self.data_folder, recursive=True))
        _all |= set(glob('**/*.png', root_dir=self.data_folder, recursive=True))
        for pattern in self.ignore_patters:
            _all -= set(glob(pattern, root_dir=self.data_folder, recursive=True))
        return _all
    
    def get_unannotated_image(self) -> PathLike:
        """The queue is randomly shuffled, so the first element is a random unannotated image."""
        return self.not_annotated_queue[0]

    def _display_image(self, path: PathLike):
        """Display the image with the given image_id on self.annotate_ax and self.annotate_fig."""
        path = os.path.join(self.data_folder, path)
        ax = self.annotate_ax
        fig = self.annotate_fig
        ax.clear()
        _, image_id, _ = split_path(path)
        img = Image.open(path)
        ax.imshow(img)  
        # Show image name 
        ax.set_title(image_id)
        
        num_annotations = len(self.annotated_set)
        n_images = len(self.all_paths)
        # Clear the figure to remove previous annotation count text
        fig.texts.clear()
        # Show number of annotated images at the top of the figure
        fig.text(0.95, 0.98, f"Annotated: {num_annotations} ({num_annotations/n_images*100:.2f}%)", verticalalignment='top', horizontalalignment='right', fontsize=10)
        # show percentage of annotated images below

        fig.canvas.draw()  

    def _on_key(self, event):
        """Function to be called when a key is pressed."""
        # Check if the key pressed is a valid annotation
        if event.key in self.keys:
            # Append the annotation to the CSV file
            label = self.key_to_label_map[event.key]
            self.append_annotation_to_csv(self.tmp_image_path, label)

            # Check if there are more images to annotate
            if self.not_annotated_queue: 
                # Pick a new image to annotate
                self.tmp_image_path = self.get_unannotated_image() 
                # Display the new image
                self._display_image(self.tmp_image_path) 
            else:
                # Close the figure if there are no more images
                plt.close() 
        elif event.key == 'escape':
            print("Escape key pressed, closing the figure")
            plt.close()
        else:
            print(self.key_err_msg)

    def annotate_images(self):
        """Start annotating the images."""
        # Check if there are images to annotate
        if self.not_annotated_queue:
            self.tmp_image_path = self.get_unannotated_image()  # Get the first image to annotate
            fig, ax = plt.subplots()
            self.annotate_ax = ax
            self.annotate_fig = fig
            # Connect the event to the function
            fig.canvas.mpl_connect('key_press_event', self._on_key) 
            # Display the first image
            self._display_image(self.tmp_image_path) 
            plt.show()
        else:
            print("No images to annotate.")

    
if __name__ == "__main__":
    cfg = Config('configs/config.yaml')
    
    data_folder = cfg.get("data", "raw_path")
    output_folder = cfg.get("data", "annotations_path")

    # Ask user for name of the output csv file
    output_csv = input("Please enter your name (This will be the name of the output csv file, use same name as previously to continue labelling): ") + ".csv"
    output_csv = output_csv.replace(" ", "_").lower()

    print(f"Output csv file will be saved as {output_csv}")

    csv_path = os.path.join(output_folder, output_csv)

    annotater = DatasetAnnotation(data_folder, csv_path)
    annotater.annotate_images()


