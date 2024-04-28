import os
from os import PathLike
from glob import glob
from functools import cached_property

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import Config
from src.utils.misc import split_path, hash_image_name

from typing import cast
from pathlib import Path

import cv2

class CornerAnnotation:
    def __init__(self, data_folder: Path, output_csv: PathLike, *, ignore_patters: list[str] | None=None):
        
        self.data_folder : Path = data_folder
        self.output_csv = output_csv
        self.columns = ['image', 'path', 'corners', 'discard', 'hash']
        self.ignore_patters = ignore_patters or ['**/never/**']
        
        self.tmp_image_path : Path = Path()

        # Make csv file if it does not exist
        if not os.path.exists(self.output_csv):
            # make dir
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.output_csv, index=False)
        
        df = pd.read_csv(self.output_csv)
        self.annotated_set = set(df['path'])
        
        self.tmp_corners = []

    def append_annotation_to_csv(self, path: PathLike, corners: list, discard : bool):
        """Append the annotation to the CSV file."""
        # Append the annotation to the CSV file'
        _, image_id, _ = split_path(path)
        row = [image_id, (self.data_folder / path).relative_to('data'), corners, discard, hash_image_name(image_id)]
        df = pd.DataFrame([row], columns=self.columns)
        df.to_csv(self.output_csv, mode='a', index=False, header=not os.path.exists(self.output_csv))
        self.annotated_set.add(path)

    @cached_property
    def all_paths(self) -> set[str]:
        """Return all the image paths in the data folder."""
        _all = set(glob('**/*.jpg', root_dir=self.data_folder, recursive=True))
        _all |= set(glob('**/*.png', root_dir=self.data_folder, recursive=True))
        for pattern in self.ignore_patters:
            _all -= set(glob(pattern, root_dir=self.data_folder, recursive=True))
        return _all
    
    def _setup_fig(self):
        
        self.fig = plt.figure()
        
        # Connect the event to the function
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
         
    def _display_image(self, path: PathLike):
        """Display the image with the given image_id on self.annotate_ax and self.annotate_fig."""
        path = Path(self.data_folder) / path
        
        _, image_id, _ = split_path(path)
        img = cv2.imread(str(path))
        
        if img is None:
            print(f"Could not open or read image file: {path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)  
        # Show image name 
        plt.title(f"image: {image_id}, Ann. corners. Topleft first, then clockwise")
        
        num_annotations = len(self.annotated_set)
        n_images = len(self.all_paths)
        # Clear the figure to remove previous annotation count text
        self.fig.texts.clear()
        # Show number of annotated images at the top of the figure
        self.fig.text(0.95, 0.98, f"Annotated: {num_annotations} ({num_annotations/n_images*100:.2f}%)", verticalalignment='top', horizontalalignment='right', fontsize=10)
        # show percentage of annotated images below

        self.fig.canvas.draw()
        plt.show(block=True)

    def _on_key(self, event):
        """Function to be called when a key is pressed."""
        # Check if the key pressed is a valid annotation
        if (event.key == 'enter' or event.key == 'space' or event.key == 'd') and self.tmp_image_path != "":
            
            if len(self.tmp_corners) == 0:
                image = cv2.imread(str(Path(self.data_folder) / self.tmp_image_path))
                w,h = image.shape[1], image.shape[0]
                w,h = round(w,2), round(h,2)
                self.tmp_corners = [[0,0], [w,0], [w,h], [0,h]]
            elif len(self.tmp_corners) != 4:
                print("Please select all 4 corners or none")
                return
            
            discard = True if event.key == 'd' else False

            # Append the annotation to the CSV file
            self.append_annotation_to_csv(self.tmp_image_path, self.tmp_corners, discard)
            
            self.tmp_image_path = Path()
            self.tmp_corners = []
            
            plt.close('all')
            
        elif event.key == 'r':
            if len(self.tmp_corners) > 0:
                self.tmp_corners.pop()
                plt.clf()
                self._display_image(self.tmp_image_path)
                for i, corner in enumerate(self.tmp_corners):
                    p1 = (cast(float,corner[0]),cast(float,corner[1]))
                    if i > 0:
                        p2 = (cast(float,self.tmp_corners[i-1][0]),cast(float,self.tmp_corners[i-1][1]))
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-') # draw line
                    plt.plot(*p1, 'ro')
                self.fig.canvas.draw()
        elif event.key == 'escape' or event.key == 'q':
            print("Escape key pressed, closing the figure")
            plt.close('all')
        else:
            print("Invalid key pressed: ", event.key)
    
    def _on_move(self, event):
        """Function to be called when the mouse is moved."""
        if event.inaxes:
            self.tmp_mouse_pos = [event.xdata, event.ydata]
        else:
            self.tmp_mouse_pos = None
    
    def _on_click(self, event):
        
        if self.tmp_mouse_pos is not None and event.button == 1: # left click
            if len(self.tmp_corners) < 4:
                self.tmp_corners.append(self.tmp_mouse_pos)
                p1 = (cast(float,self.tmp_mouse_pos[0]), cast(float,self.tmp_mouse_pos[1]))
                if len(self.tmp_corners) > 1:
                    p2 = (cast(float,self.tmp_corners[-2][0]), cast(float,self.tmp_corners[-2][1]))
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-') # draw line
                plt.plot(*p1, 'ro')
                self.fig.canvas.draw()
            else:
                print("All corners have been selected, press enter to save the annotation")
        else:
            self.tmp_mouse_pos = None
    

    def annotate_image(self, image_path: Path) -> int:
        """Start annotating the images."""
        # Check if there are images to annotate
        self._setup_fig()
        self.tmp_image_path = image_path
        
        # Display the first image
        self._display_image(self.tmp_image_path) 
        return 1

    
if __name__ == "__main__":
    cfg = Config('configs/config.yaml')
    
    data_folder = Path(cast(str,cfg.get("data", "raw_path")))
    output_folder = Path(cast(str,cfg.get("data", "elo_annotations_path")))

    # Ask user for name of the output csv file
    output_csv = "corners.csv"

    print(f"Output csv file will be saved as {output_csv}")

    csv_path = output_folder / output_csv

    annotater = CornerAnnotation(data_folder, csv_path)
    
    for path in annotater.all_paths:
        if path not in annotater.annotated_set:
            annotater.annotate_image(Path(path))


