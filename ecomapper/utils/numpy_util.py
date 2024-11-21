"""
Basic numpy tricks.
"""

import numpy as np


def join_splits_horizontal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Horizontally joins the right half of ``a`` with the left half of ``b``.

    :param a: First array.
    :param b: Second array.
    :return: Merged array.

    Examples
    --------
    >>> join_splits_horizontal(np.array([[1, 2], [3, 4],
    ... [5, 6], [7, 8]]), np.array([[9, 10], [11, 12],
    ... [13, 14], [15, 16]]))
    array([[ 2,  9],
           [ 4, 11],
           [ 6, 13],
           [ 8, 15]])
    """
    assert len(a) == len(b)
    mid = a.shape[1] // 2
    return np.hstack([a[:, mid:], b[:, :mid]])


def join_splits_vertical(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vertically joins the bottom half of ``a`` with the top half of ``b``.

    :param a: First array.
    :param b: Second array.
    :return: Merged array.

    >>> join_splits_vertical(np.array([[1, 2], [3, 4],
    ... [5, 6], [7, 8]]), np.array([[9, 10], [11, 12],
    ... [13, 14], [15, 16]]))
    array([[ 5,  6],
           [ 7,  8],
           [ 9, 10],
           [11, 12]])
    """
    mid = a.shape[0] // 2
    return np.vstack([a[mid:], b[:mid]])
