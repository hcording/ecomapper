"""
Tqdm progress bar utilities.
"""


def get_bar_format() -> str:
    """
    Returns the progress bar format used to get consistent bar lengths.

    :return: Progress bar format string.
    """
    return "{desc}: {percentage:3.0f}%|{bar:10}{r_bar}"
