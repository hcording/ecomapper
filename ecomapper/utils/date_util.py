"""
Date formatting and conversion.
"""

import datetime
from typing import Literal


def get_filename_friendly_date_format() -> Literal["%b_%d_%Y-%H_%M_%S"]:
    """
    Date format suitable for filenames, without spaces or special characters.

    :return: Date with format "Jan_01_2023-03_38_59".

    Examples
    --------
    >>> get_filename_friendly_date_format()
    '%b_%d_%Y-%H_%M_%S'
    """
    return "%b_%d_%Y-%H_%M_%S"


def get_humanly_readable_date_format() -> Literal["%d %b %Y at %H:%M:%S"]:
    """
    Easily readable date format.

    :return: Date with format "Jan 01 2023 at 12:05:07".

    Examples
    --------
    >>> get_humanly_readable_date_format()
    '%d %b %Y at %H:%M:%S'
    """
    return "%d %b %Y at %H:%M:%S"


def filename_friendly_date() -> str:
    """
    Gets the current date and time without spaces or special characters.

    :return: Date with format "Jan_01_2023-03_38_59".
    """
    return datetime.datetime.now().strftime(
        get_filename_friendly_date_format())


def convert_to_humanly_readable_date(date: str) -> str:
    """
    Converts a date in filename-friendly format to humanly readable format.

    :param date: Date with format "%b_%d_%Y-%H_%M_%S".
    :return: Date with format "%d %b %Y at %H:%M:%S".
    """
    return datetime.datetime.strptime(
        date,
        get_filename_friendly_date_format()).strftime(
        get_humanly_readable_date_format())
