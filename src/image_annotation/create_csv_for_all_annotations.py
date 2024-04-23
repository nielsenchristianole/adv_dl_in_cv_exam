import os
from typing import Optional, List
from glob import glob

import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.misc import split_path, hash_image_name


def main(label_to_exclude_search_strings: Optional[dict[str, str]]=None) -> None:
    label_to_exclude_search_strings = {'never': '**/never/**'} if label_to_exclude_search_strings is None else label_to_exclude_search_strings
    assert 'default' not in label_to_exclude_search_strings, 'default is a reserved label name. Please choose another name.'
    columns = ['image', 'path', 'label', 'hash']

    cfg = Config('configs/config.yaml')

    root_dir = cfg.get('data', 'raw_path')
    out_path = cfg.get('data', 'annotations_path')

    all_paths = set(glob('**/*.jpg', root_dir=root_dir, recursive=True))
    all_paths |= set(glob('**/*.png', root_dir=root_dir, recursive=True))

    label_to_image_set = dict()
    for label, pattern in label_to_exclude_search_strings.items():
        label_set = set(glob(pattern, root_dir=root_dir, recursive=True)) & all_paths
        all_paths -= label_set
        label_to_image_set[label] = label_set
    label_to_image_set['default'] = all_paths

    dfs = []
    for label, image_set in label_to_image_set.items():
        image_set = list(image_set)
        image_ids = [split_path(p)[1] for p in image_set]
        labels = len(image_set) * [label]
        hashes = [hash_image_name(image_id) for image_id in image_ids]
        data = np.array((image_ids, image_set, labels, hashes)).T

        df = pd.DataFrame(data, columns=columns)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(os.path.join(out_path, 'all_wanted_images.csv'), index=False)


if __name__ == '__main__':
    main()