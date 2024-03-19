import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tkinter as tk

from src.utils.config import Config



class DatasetAnnotation:
    def __init__(self, data_folder, output_csv):
        self.data_folder = data_folder
        self.output_csv = output_csv
        # Make csv file if it does not exist
        if not os.path.exists(self.output_csv):
            # make dir
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            df = pd.DataFrame(columns=['ImageID', 'Label'])
            df.to_csv(self.output_csv, index=False)
    
    def append_annotation_to_csv(self, image_id, label):
        """Append the annotation to the CSV file."""
        # Append the annotation to the CSV file
        df = pd.DataFrame([[image_id, label]], columns=['ImageID', 'Label'])
        df.to_csv(self.output_csv, mode='a', index=False, header=not os.path.exists(self.output_csv))

    @property
    def all_ids(self):
        """Return all the image ids in the data folder."""
        return os.listdir(self.data_folder)
    
    @property
    def image_ids_not_annotated(self):
        """Return the image ids that have not been annotated yet."""
        # Check which images have not been annotated yet
        if not os.path.exists(self.output_csv):
            return self.image_ids
        else:
            annotated_ids = pd.read_csv(self.output_csv)['ImageID'].tolist()
            return [image_id for image_id in self.all_ids if image_id not in annotated_ids]
        
    def get_random_image(self):
        images_not_annotated = self.image_ids_not_annotated
        # get random idx 
        idx = np.random.randint(0, len(images_not_annotated))
        return self.image_ids_not_annotated[idx]

    def _display_image(self, image_id):
        """Display the image with the given image_id on self.annotate_ax and self.annotate_fig."""
        ax = self.annotate_ax
        fig = self.annotate_fig
        ax.clear()  
        image_path = os.path.join(self.data_folder, image_id)
        img = Image.open(image_path)
        ax.imshow(img)  
        # Show image name 
        ax.set_title(image_id)
        fig.canvas.draw()  

    def _on_key(self, event):
        """Function to be called when a key is pressed."""
        # Check if the key pressed is a valid annotation
        if event.key in ['1', '2', '3', '4']:
            # Append the annotation to the CSV file
            self.append_annotation_to_csv(self.image_id, event.key)

            # Check if there are more images to annotate
            if self.image_ids_not_annotated: 
                # Pick a new image to annotate
                self.image_id = self.get_random_image() 
                # Display the new image
                self._display_image(self.image_id) 
            else:
                # Close the figure if there are no more images
                plt.close() 
        elif event.key == 'escape':
            print("Escape key pressed, closing the figure")
            plt.close()
        else:
            print("Invalid key pressed, please press '1', '2', '3', or '4'")

    def annotate_images(self):
        """Start annotating the images."""
        # Check if there are images to annotate
        if self.image_ids_not_annotated:
            self.image_id = self.get_random_image()  # Get the first image to annotate
            fig, ax = plt.subplots()
            self.annotate_ax = ax
            self.annotate_fig = fig
            # Connect the event to the function
            fig.canvas.mpl_connect('key_press_event', self._on_key) 
            # Display the first image
            self._display_image(self.image_id) 
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


