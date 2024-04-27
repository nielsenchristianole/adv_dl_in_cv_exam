import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.misc import hash_image_name
from src.utils.config import Config


NUM_MIN_GAMES = 5
ELO_THRESHOLD_QUANTILE = 0.8
FIX_CSV_NAME = 'calle2.csv'


cfg = Config('configs/config.yaml')
in_dir = cfg.get('data', 'elo_annotations_path')
out_dir = cfg.get('data', 'annotations_path')

# load
df_calle: pd.DataFrame = pd.read_csv(os.path.join(in_dir, FIX_CSV_NAME))
df_corners = pd.read_csv(os.path.join(in_dir, 'corners.csv'))

# discard
df_calle = df_calle[df_calle['discard'] == False]
df_corners = df_corners[df_corners['discard'] == False]

# correct column and format
df_calle['image'] = [str(Path(p).stem) for p in df_calle['name']]

# filter out non cornered images
known = set(df_corners['image'])
mask = np.array([image in known for image in df_calle['image']])
mask &= df_calle['games'] >= 5

df_calle = df_calle[mask]

# add rest of columns
df_calle['path'] = [p.split('data\\')[-1] for p in df_calle['path']]
df_calle['hash'] = [hash_image_name(n) for n in df_calle['image']]

elo_threshold = df_calle['elo'].quantile(ELO_THRESHOLD_QUANTILE)
df_calle['label'] = (df_calle['elo'] >= elo_threshold) + 0

# save
df_calle = df_calle[['image', 'path', 'label', 'hash']]
df_calle.to_csv(
    os.path.join(out_dir, 'calle2.csv'),
    index=False
)
