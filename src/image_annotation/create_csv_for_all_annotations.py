import os
from os import PathLike
from glob import glob

import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.misc import split_path, hash_image_name


def main() -> None:
    exclude_search_strings = ['**/never/**']
    columns = ['image', 'path', 'label', 'hash']

    cfg = Config('configs/config.yaml')

    root_dir = cfg.get('data', 'raw_path')
    out_path = cfg.get('data', 'annotations_path')

    all_paths = set(glob('**/*.jpg', root_dir=root_dir, recursive=True))
    all_paths |= set(glob('**/*.png', root_dir=root_dir, recursive=True))

    for pattern in exclude_search_strings:
        all_paths -= set(glob(pattern, root_dir=root_dir, recursive=True))

    all_paths = list(all_paths)
    image_ids = [split_path(p)[1] for p in all_paths]
    labels = len(all_paths) * ['default']
    hashes = [hash_image_name(image_id) for image_id in image_ids]
    data = np.array((image_ids, all_paths, labels, hashes)).T

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(out_path, 'all_wanted_images.csv'), index=False)


if __name__ == '__main__':
    main()