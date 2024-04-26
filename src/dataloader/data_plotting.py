
from pathlib import Path
from src.utils.config import Config
from typing import cast
import pandas as pd
import numpy as np
import ast
import cv2
import matplotlib.pyplot as plt
import tqdm

def crop_and_wrap(img, corners):
    
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

if __name__ == "__main__":
    
    cfg = Config(Path("configs/config.yaml"))
    
    anno_folder = Path(cast(str,cfg.get("data", "elo_annotations_path")))
    
    annotations = pd.read_csv(anno_folder / "corners.csv")
    corners_raw = annotations['corners']
    paths = annotations['path']
        
    # remove all paths and corners that has "discard"== True
    discard = annotations['discard']
    corners_raw = corners_raw[discard == False]
    img_paths = list(paths[discard == False])
    
    corners = []
    for corner in corners_raw:
        corners.append(ast.literal_eval(corner))
        
    
    import concurrent.futures

    def process_image(idx):
        crn = np.array(corners)[idx]
        
        img = cv2.imread("data/" + img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        color = (255, 0, 0)
        
        img_annotated = img.copy()
        
        img = crop_and_wrap(img, crn)
        
        plt.imsave(f"data/cropped/{Path(img_paths[idx]).stem}.png", img)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx in tqdm.tqdm(range(len(corners))):
            future = executor.submit(process_image, idx)
            futures.append(future)
        
        # Wait for all threads to complete
        concurrent.futures.wait(futures)