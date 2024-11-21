"""
JSON extensions using ``jsonpickle``.
"""

import importlib
import json
from json import JSONDecodeError
from typing import Any

from ecomapper.utils.file_util import atomic_write
from ecomapper.utils.path_util import expand_path
from ecomapper.utils.print_util import warn


def try_deserialize_json_file(path: str,
                              verbose=True) -> dict | None:
    """
    Attempts to deserialize the JSON file at ``path`` using ``jsonpickle``.
    If deserialization fails, ``None`` is returned.

    :param path: Filepath to load from.
    :param verbose: If ``True``, messages are printed to describe any errors
        during deserialization.
    :return: The deserialized object as ``dictionary``, or ``None`` if loading
        failed.
    """
    if path == '':
        return None

    path = expand_path(path)
    try:
        with open(path, 'r') as f:
            return json.load(f, object_pairs_hook=ensure_unique_keys)
    except JSONDecodeError as exc:
        if verbose:
            warn("Error parsing JSON file")
            print(exc)
        return None
    except DuplicateKeyError:
        if verbose:
            warn("JSON file contains duplicate keys")
        return None
    except FileNotFoundError as exc:
        if verbose:
            warn("File not found")
            print(exc)
        return None
    except BaseException as exc:
        if verbose:
            warn("File is not a valid JSON file")
            print(exc)
        return None


class DuplicateKeyError(Exception):
    """
    Exception raised when loading a JSON file with duplicate keys.
    """
    pass


def ensure_unique_keys(pairs: list[tuple[Any, Any]]) -> dict:
    """
    Checks that the given key-value pairs have unique keys.

    :param pairs: Key-value pairs to check.
    :return: The loaded ``dict``.
    :raises DuplicateKeyError: If there are duplicate keys in ``pairs``.
    """
    d = {}
    for k, v in pairs:
        if k in d:
            raise DuplicateKeyError(f"Duplicate key: {k}")
        else:
            d[k] = v
    return dict(pairs)


def save_to_json(obj: Any, path: str) -> None:
    """
    Saves the object as JSON to a given file path.

    :param obj: The object to be saved.
    :param path: The location to save to.
    """
    jsonpickle = importlib.import_module('jsonpickle')
    atomic_write(jsonpickle.encode(obj, indent=4), path, mode='w+')
