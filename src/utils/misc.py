import os
import yaml
import hashlib
from os import PathLike


def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def split_path(path: PathLike) -> tuple[str, str, str]:
    _dir, filename = os.path.split(path)
    image_id, ext = os.path.splitext(filename)
    return _dir, image_id, ext


def hash_image_name(image_name: str) -> float:
    """
    Returns a float in the range [0, 1). Used for deterministic data splitting.
    """
    precision = 0xffffffff
    _hash = hashlib.sha256(image_name.encode(), usedforsecurity=False).digest()
    _int = int.from_bytes(_hash, 'little')
    normalized = (_int % precision) / precision
    return normalized
    
