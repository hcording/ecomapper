"""
Path manipulation.
"""

import os
from pathlib import Path


def expand_path(path: str) -> str:
    """
    Unwraps the home character (~) in the path (if present), and converts the
    path to an absolute path.

    :param path: Path to expand.
    :return: Expanded path.
    """
    return str(Path(path).expanduser().resolve())


def make_dir(path: str, exists_ok: bool = True, require_empty=False) \
        -> str:
    """
    Creates a directory on the given path.

    :param path: Path to create directory at.
    :param exists_ok: Whether it is acceptable that the directory
        already exists.
    :param require_empty: Whether the directory must be empty if it already
        exists.
    :return: Path to the created directory.
    """
    assert path != '', "Path is empty"

    path = expand_path(path)

    if require_empty and os.path.exists(path):
        if len(os.listdir(path)) != 0:
            raise RuntimeError(f"Directory '{path}' exists and is not empty")

    os.makedirs(path, exist_ok=exists_ok)
    return path


def get_stems(directory: str):
    """
    Returns the stems of all files in ``directory``.
    Stems are filenames without extension, e.g., "/path/to/my/image1.jpg" has
    stem "image1".

    :param directory: Directory to scan.
    :return: File stems.
    """
    return sorted((Path(x).stem for x in os.listdir(directory)))
