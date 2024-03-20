import yaml
import os
from os import PathLike


def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def split_path(path: PathLike) -> tuple[str, str, str]:
    _dir, filename = os.path.split(path)
    image_id, ext = os.path.splitext(filename)
    return _dir, image_id, ext


def image_path_to_endocing_path(image_path: str, encoding_base_path: str = 'endodings') -> str:
    with_other_extension = os.path.splitext(image_path)[0] + '.npy'
    encoding_path = os.path.join(encoding_base_path, with_other_extension)
    return encoding_path