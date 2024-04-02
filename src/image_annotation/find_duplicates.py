
import ast
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import tqdm
from functools import cache, partial

import hashlib
from scipy.spatial import cKDTree

def embed(img):

    f1 = np.mean(img[:,:,0])
    f2 = np.mean(img[:,:,1])
    f3 = np.mean(img[:,:,2])
    
    # Convert to another color space
    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    f4 = np.mean(LAB[:,:,0])
    f5 = np.mean(LAB[:,:,1])
    f6 = np.mean(LAB[:,:,2])
    
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    f7 = np.mean(HSV[:,:,0])
    f8 = np.mean(HSV[:,:,1])
    f9 = np.mean(HSV[:,:,2])
    
    YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    f10 = np.mean(YUV[:,:,0])
    f11 = np.mean(YUV[:,:,1])
    f12 = np.mean(YUV[:,:,2])
    
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    
@cache
def crop_and_wrap(img_path, corner_path):
    
    img = cv2.imread(f"data/{str(img_path)}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    corners = pd.read_csv(corner_path)
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
    
def fast_knn(features, k=5):
    tree = cKDTree(features)
    distances, indices = tree.query(features, k=k+1)
    return indices[:, 1:], distances[:, 1:]

def discard_if_duplicate(event, i, j, corners, corner_path):
    if event.key == 'y':
        
        path1 = Path(corners['path'][i])
        path2 = Path(corners['path'][j])
        corner_path = Path(corner_path)
        
        elo = pd.read_csv(corner_path.parent.parent / f"elo_annotations/{corner_path.stem}.csv")
        
        games1 = elo[elo['name'] == str(path1)]['games'].values[0]
        games2 = elo[elo['name'] == str(path2)]['games'].values[0]
        
        if games1 > games2:
            corners.loc[j, 'discard'] = True
            corners.loc[i, 'discard'] = False
        else:
            corners.loc[j, 'discard'] = False
            corners.loc[i, 'discard'] = True
        
        corners.to_csv(corner_path, index=False)
        print(f"Discarded {img_paths[i]} and {img_paths[j]}")
        plt.close()
    elif event.key == 'q':
        print(f"Skipped {img_paths[i]} and {img_paths[j]}")
        plt.close()

if __name__ == "__main__":
    
    corner_path = Path(f"data/corner_annotations/corners.csv")
    
    corners = pd.read_csv(corner_path)
    img_paths = corners['path']
    discards = corners['discard']
    print("Images with corners annotated: ", len(img_paths))
    
    
    features = []
    for i, (img_path, discard) in tqdm.tqdm(enumerate(zip(img_paths, discards))):
        img_path = Path(img_path)
        
        img = crop_and_wrap(img_path, corner_path)

        features.append(embed(img)) # type: ignore
        
    features = np.array(features)
    
    arg_dist, dist = fast_knn(features)
    
    #free the memory
    del features
    
    threshold = 3
    duplicates = []
    for i in range(len(arg_dist)):
        for j in range(len(arg_dist[i])):
            if dist[i][j] < threshold:
                duplicates.append((i, arg_dist[i][j]))
                
    del arg_dist, dist
                
    print("Number of duplicates: ", len(duplicates))
    
    duplicates = np.array(duplicates)
    
    for i, j in duplicates:
        
        corners = pd.read_csv(corner_path)
        img_paths = corners['path']
        discards = corners['discard']
            
        if discards[i] or discards[j]:
            continue
        
        print(f"{img_paths[i]} and {img_paths[j]}")
        
        # Show the images
        img1 = crop_and_wrap(Path(img_paths[i]), corner_path) # type: ignore
        img2 = crop_and_wrap(Path(img_paths[j]), corner_path) # type: ignore
        
        fig, ax = plt.subplots(1, 2)
        
        func = partial(discard_if_duplicate, i=i, j=j, corners=corners, corner_path=corner_path)
        
        fig.canvas.mpl_connect('key_press_event', func)
        
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        plt.show()
    
    