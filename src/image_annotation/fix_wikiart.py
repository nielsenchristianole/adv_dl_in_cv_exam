import os

import numpy as np
import pandas as pd

from src.utils.misc import hash_image_name

root_dir = 'data/wikiart/'

raw_df = pd.read_csv('data/annotations/wikiart_raw.csv')

df = raw_df[raw_df['subset'] != 'uncertain artist']
df = df[df['genre_count'] == 1]
exists = np.array([os.path.exists(os.path.join(root_dir, p)) for p in df['filename']])
df = df[exists]

df['image'] = [os.path.splitext(os.path.split(p)[1])[0] for p in df['filename']]
df['hash'] = [hash_image_name(p) for p in df['image']]
df['path'] = df['filename']
df['label'] = [l.removeprefix('[\'').removesuffix('\']') for l in df['genre']]

df_train = df[df['subset'] == 'train']
df_test = df[df['subset'] == 'test']

keep_columns = ['image', 'path', 'label', 'hash']

df[keep_columns].to_csv('data/annotations/wikiart_all.csv', index=False)
df_train[keep_columns].to_csv('data/annotations/wikiart_train.csv', index=False)
df_test[keep_columns].to_csv('data/annotations/wikiart_test.csv', index=False)
