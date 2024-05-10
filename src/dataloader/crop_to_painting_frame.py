import ast
from typing import cast
from pathlib import Path
from functools import partial

import cv2
import numpy as np
import pandas as pd
import concurrent.futures

from src.utils.config import Config


def crop_and_wrap(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    
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


def process_image_factory(idx: int, corners: np.ndarray, img_paths: np.ndarray, img_root_dir: Path, crop_root_dir: Path) -> None:
    crn = corners[idx]
    img_path: str = img_paths[idx]
    
    img = cv2.imread(str(img_root_dir / img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_and_wrap(img, crn)

    out_path = crop_root_dir / img_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():

    cfg = Config(Path("configs/config.yaml"))
    anno_folder = Path(cast(str, cfg.get("data", "elo_annotations_path")))

    annotations = pd.read_csv(anno_folder / "corners.csv")
    img_root_dir = Path(cast(str, cfg.get('data', 'raw_path')))
    crop_root_dir = Path(cast(str, cfg.get('data', 'cropped_path')))
    
    keep = ~ annotations['discard'].to_numpy()

    corners_raw = annotations['corners'][keep]
    img_paths = annotations['path'][keep].to_numpy()
    
    corners = np.stack(corners_raw.apply(ast.literal_eval))

    process_image = partial(process_image_factory, corners=corners, img_paths=img_paths, img_root_dir=img_root_dir, crop_root_dir=crop_root_dir)
    process_image(0)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx in range(len(corners)):
            future = executor.submit(process_image, idx)
            futures.append(future)
        
        # Wait for all threads to complete
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
