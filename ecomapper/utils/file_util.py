"""
Files and atomic I/O.
"""

import os
import os.path
import shutil
import tempfile
import zipfile
from glob import glob
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable


def file_to_list(path: str) -> list[str]:
    """
    Writes each line of a file into a list.

    :param path: Path pointing to file.
    :return: List with file contents.
    """
    assert os.path.isfile(path)

    result = []
    with open(path) as f:
        while line := f.readline():
            stripped_line = line.rstrip()
            if stripped_line != '':
                result.append(stripped_line)
    return result


def list_to_file(items: Iterable[str], file: str, delim='\n') -> None:
    """
    Writes a list to a file, separating list items by ``delim``.
    :param delim: Delimiter for list items written to the file.
    :param items: List of items to write.
    :param file: Filepath to write to.
    """
    with open(file, 'w') as f:
        f.write(delim.join(items))


def atomic_write(data: str, path: str, mode: str) -> None:
    """
    Writes atomically to a file, if the file system supports atomic file
    renaming.

    :param data: Data to write.
    :param path: Filepath to write to.
    :param mode: Mode (cannot be a readonly mode)
    """

    def write(d, p):
        with open(p, mode) as f:
            f.write(d)

    atomic_action(data, path, write)


def atomic_action(data: Any, filepath: str,
                  action: Callable[[Any, str], Any]) -> None:
    """
    Performs the given ``action``, which must produce a file.
    This operation is atomic where possible, using ``os.rename``. The
    ``action`` result is written to a temporary file, which is then moved to
    the given ``file`` path.

    :param data: Data to pass to ``action``.
    :param filepath: Final path for the output of ``action``.
    :param action: A callable which takes ``data`` and writes a
        file to the given temporary ``path``.

    Notes
    -----
    The temporary file will be created in the parent directory of ``file``.
    """
    assert not os.path.isdir(filepath), \
        "Save path cannot be a directory"

    temp_file_fd, temp_file_path = None, None
    try:
        ext = os.path.splitext(filepath)[1]
        temp_file_fd, temp_file_path = \
            tempfile.mkstemp(dir=Path(filepath).parent, suffix=ext)
        action(data, temp_file_path)
        shutil.move(temp_file_path, filepath)
    finally:
        if temp_file_fd:
            os.close(temp_file_fd)
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def get_unique_file_in_directory(directory: str,
                                 pattern: str,
                                 predicate: Callable[[str], bool] | None = None) -> str | None:
    """
    Gets a unique file from a directory, returning ``None`` if the file does
    not exist, and raising a ``RuntimeError`` if the file is not unqiue.

    :param directory: Directory to scan.
    :param pattern: Pattern to match filenames on.
    :param predicate: Additional criteria to apply to search results.
    :return: Unique filepath, or None if no file was found.
    :raises RuntimeError: If more than one file matching ``pattern`` exists in
        ``directory``.
    """
    files = glob(os.path.join(directory, pattern))
    if predicate:
        files = [x for x in files if predicate(x)]

    if len(files) > 1:
        raise RuntimeError(f"Found multiple files with pattern {pattern} "
                           "but expected at most one file")

    return files[0] if len(files) == 1 else None


def extract_folder_from_zip(zip_file: str,
                            output_dir: str,
                            folder_name: str):
    """
    Extracts a specific folder from a zip file and places it in a given
    location.

    :param zip_file: Path to the ZIP file.
    :param output_dir: Directory where the extracted folder should be placed.
    :param folder_name: Name of the folder to extract.
    """
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        done = False
        for file in zip_ref.namelist():
            if file.startswith(folder_name + '/'):
                zip_ref.extract(file, output_dir)
                done = True
        if not done:
            raise RuntimeError(
                f"Given folder not contained in zip file: {folder_name}")
