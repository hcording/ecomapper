"""
Wrappers for reading and writing with ``joblib``.
"""

from typing import Any
from typing import Type
from typing import TypeVar

import joblib

from ecomapper.utils.file_util import atomic_action

T = TypeVar('T')


def joblib_load_typed(filename: Any, class_type: Type[T],
                      mmap_mode: Any = None) -> T:
    """
    Loads an object from disk using ``joblib`` and asserts that the loaded
    object's type is ``class_type``.

    :param filename: Filepath to load from.
    :param class_type: Expected type.
    :param mmap_mode: Memory mapping mode (optional).
    :return:
    """
    result = joblib.load(filename, mmap_mode)
    assert isinstance(result, class_type), \
        f"Deserialized object has incorrect type: " \
        f"Expected {class_type} but got {result.__class__.__name__}"
    return result


def joblib_dump_atomic(data: Any, path: str) -> None:
    """
    Writes the given ``data`` to disk atomically using ``joblib`` and
    ``os.rename``.

    :param data: Data to write.
    :param path: Filepath to write to.
    """
    atomic_action(data, path, lambda x, y: joblib.dump(x, y, compress=True))
