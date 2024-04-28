import tqdm
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


input_dir = Path('data/samples')


images = list()
paths = list()
for _dir in input_dir.iterdir():
    files = list(_dir.iterdir())
    for file in files:
        img = np.load(file)
        images.append(img)
        paths.append(file)

images = np.stack(images)
images = np.clip(255 * (images / 2 + 0.5), 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)


for img, path in tqdm.tqdm(zip(images, paths)):
    _img = Image.fromarray(img)
    # change the second folder fom samples to sampled_images and save the images as png
    out_dir = Path(str(path).replace('samples', 'sampled_images').replace('npy', 'png')).parent
    os.makedirs(out_dir, exist_ok=True)
    _img.save(str(path).replace('samples', 'sampled_images').replace('npy', 'png'))
