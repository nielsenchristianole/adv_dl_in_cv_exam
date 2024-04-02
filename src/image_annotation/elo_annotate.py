

from pathlib import Path
from functools import cached_property, cache
from PIL import Image

import matplotlib
# use a backend that supports interactive mode
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import src.image_annotation.corner_annotate as corner_annotate
from src.utils.config import Config
import math

import cv2

class ELOAnnotate:
    
    def __init__(self, data_path : Path,
                       corner_path : Path,
                       out_path : Path,
                       only_annotated = False,
                       auto_save_every = 10):
        
        self.discard_patters = ['**/never/**']
        
        self.only_annotated = only_annotated
        
        self.corner_path = corner_path
        self.corner_annotator = corner_annotate.CornerAnnotation(data_folder, corner_path)
        
        self.data_path = data_path
        self.out_path = out_path
        self.out_matches_path = out_path.parent / (out_path.stem + "_matches.txt")
        self.auto_save_every = auto_save_every
        
        if not self.out_path.exists():
            # Create the output path and file
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(columns=['name', 'path', 'elo', 'games', 'discard'])
            df.to_csv(self.out_path, index=False)
        if not self.out_matches_path.exists():
            # Create the output path and file
            self.out_matches_path.parent.mkdir(parents=True, exist_ok=True)
            
        self.total_games = 0
        self.total_discard = 0
        
        self.annotations : dict[str,tuple[str, float,int,bool]] = self._get_annotations()
        
    def _create_display(self):
        fig, axs = plt.subplots(1,2)
        fig = fig
        axs = axs
        
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        return fig, axs
        
    def _get_annotations(self) -> dict[str,tuple[str, float,int,bool]]:
        """Return the annotations from the csv file."""
        df = pd.read_csv(self.out_path)
        annotations = dict(zip(df['name'], zip(df['path'], df['elo'], df['games'], df['discard'])))
        
        games = []
        for path in self._all_paths:
            name = path.name
            if name not in annotations:
                annotations[name] = (path, 1000, 0, False)
                games.append(0)
            else:
                self.total_games += annotations[name][2]/2
                self.total_discard += annotations[name][3]
                games.append(annotations[name][2])
                
        self.bin_games = np.append(np.bincount(games), np.zeros(10))
        
        return annotations
            
    def _save_annotations(self):
        """Save the annotations to the csv file."""
        anno_formated = [(name, *values) for name, values in self.annotations.items()]
        
        df = pd.DataFrame(anno_formated)
        df.to_csv(self.out_path, index=False, header=['name', 'path', 'elo', 'games', 'discard'])
        
    @cached_property
    def _all_paths(self) -> set[Path]:
        """Return all the image paths in the data folder."""
        _all = set(self.data_path.glob('**/*.jpg'))
        _all |= set(self.data_path.glob('**/*.png'))
        for pattern in self.discard_patters:
            _all -= set(self.data_path.glob(pattern + '/*'))
        return _all
    
    def _pick_images(self):
        """
        Pick two images to compare based on their ELO ratings.
        The probability of picking each image is inversely proportional to the number of games it has been played.
        """

        corner_annotations = pd.read_csv(self.corner_path)
        
        if self.only_annotated:
            # get all the key-value pairs where the image has been annotated in the corner annotations
            annotation_subset = {name: self.annotations[name] for name in corner_annotations['path'].values}
        else:
            annotation_subset = self.annotations

        # Calculate the probability of picking each image, inversely proportional to the number of games it has been played
        if self.total_games == 0:
            probs = {name: 1 for name in annotation_subset if not annotation_subset[name][3]}
        else:
            probs = {name: math.exp(-games) \
                            for name, (_, _, games, discard)  in annotation_subset.items() \
                            if not discard}

        # Normalize the probs so they sum to 1
        total_prob = sum(probs.values())
        probs = {name: prob / total_prob for name, prob in probs.items()}

        # Pick an image based on the probs
        names, probs = zip(*probs.items())
        names = np.random.choice(names, 2, p=probs, replace=False)
        
        # pick image two as the image with the closest ELO rating to image one
        name1 = names[0]
        name2 = names[1]
        # name2 = min(names[1:], key=lambda name: abs(annotation_subset[name][1] - annotation_subset[name1][1]))
        
        # If name1 and name2 does not exist in the corner annotations, annotate them
        if name1 not in corner_annotations['path'].values or name2 not in corner_annotations['path'].values:
            
            plt.close(self.fig)
            
            if name1 not in corner_annotations['path'].values:
                self.corner_annotator.annotate_image(name1)
            while len(plt.get_fignums()) > 0:
                plt.pause(0.1)
            if name2 not in corner_annotations['path'].values:
                self.corner_annotator.annotate_image(name2)
            while len(plt.get_fignums()) > 0:
                plt.pause(0.1)
              
            self.fig, self.axs = self._create_display()

            corner_annotations = pd.read_csv(self.corner_path)
            
        corner1 = corner_annotations[corner_annotations['path'] == name1]
        corner2 = corner_annotations[corner_annotations['path'] == name2]
        
        if corner1['discard'].values[0] or corner2['discard'].values[0]:
            if corner1['discard'].values[0]:
                self._discard(name1)
            if corner2['discard'].values[0]:
                self._discard(name2)
            
            return self._pick_images()
        
        return name1, name2
    

    def _update_elo(self, name1, name2, result):
        """
        Update the rating based on the expected and actual outcomes.
        
        result is the result of the match, 1 if the first image won, 0 if the second image won.
        """
        # write the names to the matches file
        with open(self.out_matches_path, 'a') as f: 
            f.write(f"{name1},{name2},{result}\n")
        
        _, rating1, games1, discard1 = self.annotations[name1]
        _, rating2, games2, discard2 = self.annotations[name2]

        path1 = self.data_path / name1
        path2 = self.data_path / name2
        
        expected1 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
        expected2 = 1 - expected1
        
        rating1 += 32 * (result - expected1)
        rating2 += 32 * ((1 - result) - expected2)
        
        self.total_games += 1
        
        if len(self.bin_games) < max(games1, games2) + 1:
            self.bin_games = np.append(self.bin_games, 0)
        
        self.bin_games[games1] -= 1
        self.bin_games[games2] -= 1
        self.bin_games[games1 + 1] += 1
        self.bin_games[games2 + 1] += 1
        
        if self.total_games % self.auto_save_every == 0:
            print("Auto saving annotations")
            self._save_annotations()
        
        self.annotations[name1] = (path1, rating1, games1 + 1, discard1)
        self.annotations[name2] = (path2, rating2, games2 + 1, discard2)
    
    def _crop_and_wrap(self, img, img_path):
        
        import ast
        
        corners = pd.read_csv(self.corner_path)
        corners = corners[corners['path'] == img_path.name]['corners'].values[0]
        corners = np.array(ast.literal_eval(corners))
        
        xmin = corners[:,0].min()
        xmax = corners[:,0].max()
        ymin = corners[:,1].min()
        ymax = corners[:,1].max()
        
        square_corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        
        # Wrap image and crop to the square
        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), square_corners.astype(np.float32))
        img = cv2.warpPerspective(np.array(img), M, (w, h))
        
        # Crop to the wrapped square
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        return img
        
    
    def _update_display(self):
        
        name1, name2 = self._pick_images()
        
        self.tmp_name1 = name1
        self.tmp_name2 = name2

        _, _, games1, _ = self.annotations[name1]
        _, _, games2, _ = self.annotations[name2]

        path1 = self.data_path / name1
        path2 = self.data_path / name2

        def display(ax, path: Path, games):
            
            ax.clear()
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = self._crop_and_wrap(img, path)
            
            ax.imshow(img)
            ax.set_title(path.name + f"\nGames: {games}")
        
        for ax, path, games in zip(self.axs, [path1, path2], [games1, games2]):
            display(ax, Path(path), games)
        
        self.fig.texts.clear()
        n_images = len(self._all_paths)
        self.fig.text(0.75, 0.98, f"'1': left, '2':pick \n \
                                    'q':quit and save",
                      verticalalignment='top', horizontalalignment='right', fontsize=10)
        self.fig.text(0.95, 0.98, f"Games: {self.total_games}\n \
                                    Discard: {self.total_discard}\n \
                                    Total: {len(self.annotations)}",
                      verticalalignment='top', horizontalalignment='right', fontsize=10)
        self.fig.text(0.20, 0.98, f"Bin games:\n \
                                    0:  {(self.bin_games[0]/n_images)*100:.1f}%\n \
                                    1:  {(self.bin_games[1]/n_images)*100:.1f}%\n",
                      verticalalignment='top', horizontalalignment='right', fontsize=10)
        self.fig.text(0.35, 0.98, f"2:  {(self.bin_games[2]/n_images)*100:.1f}%\n \
                                    3:  {(self.bin_games[3]/n_images)*100:.1f}%\n \
                                    4:  {(self.bin_games[4]/n_images)*100:.1f}%\n",
                      verticalalignment='top', horizontalalignment='right', fontsize=10)
        self.fig.text(0.50, 0.98, f"5:  {(self.bin_games[5]/n_images)*100:.1f}%\n \
                                    6:  {(self.bin_games[6]/n_images)*100:.1f}%\n \
                                    7+: {(sum(self.bin_games[7:])/n_images)*100:.1f}%\n",
                      verticalalignment='top', horizontalalignment='right', fontsize=10)
        
        self.fig.canvas.draw()
        plt.show(block=True)
        
    def _discard(self, name):
        path, elo, games, discard = self.annotations[name]
        self.annotations[name] = (path, elo, games, True)
        self.total_discard += 1
    
    def _on_key(self, event):
        """Function to be called when a key is pressed."""
        # Check if the key pressed is a valid annotation
        
        match event.key:
            case '1':
                self._update_elo(self.tmp_name1, self.tmp_name2, 1)
            case '2':
                self._update_elo(self.tmp_name1, self.tmp_name2, 0)
            case '3':
                self._update_elo(self.tmp_name1, self.tmp_name2, 0.5)
            case 'q':
                print("Escape key pressed, closing and saving")
                plt.close('all')
            case _:
                print("Invalid key pressed: ", event.key)
        
        if event.key in ['1', '2', '3','q']:
            self._update_display()
                
    def annotate(self):
        
        self.fig, self.axs = self._create_display()
        self._update_display()
        self._save_annotations()

    
if __name__ == "__main__":
    cfg = Config('configs/config.yaml')
    
    data_folder = Path(str(cfg.get("data", "raw_path")))
    output_folder = Path(str(cfg.get("data", "elo_annotations_path")))

    # Ask user for name of the output csv file
    output_csv = input("Please enter your name (This will be the name of the output csv file, use same name as previously to continue labelling): ") + ".csv"
    output_csv = output_csv.replace(" ", "_").lower()

    print(f"Output csv file will be saved as {output_csv}")

    annotater = ELOAnnotate(data_folder,
                            output_folder / "corners.csv",
                            output_folder / output_csv,
                            only_annotated = True)
    annotater.annotate()


