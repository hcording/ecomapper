"""
Printing and colored text.
"""
import os
import subprocess

from termcolor import cprint


class CommonMessages:
    """
    Class to group messages that get printed often.
    """
    RETRY = "please try again"
    INVALID_INPUT_RETRY = f"Invalid input, {RETRY}"
    PATH_TO = "Please provide the path to the"


def err(message: str) -> None:
    """
    Prints a given message with red background and black text.

    :param message: Message to print.
    """
    cprint(message, "black", "on_red")


def warn(message: str) -> None:
    """
    Prints a given message with yellow background and black text.

    :param message: Message to print.
    """
    cprint(message, "black", "on_yellow")


def success(message: str) -> None:
    """
    Prints a given message with green background and black text.

    :param message: Message to print.
    """
    cprint(message, "black", "on_green")


def clear() -> None:
    """
    Clears the screen in the terminal.
    """
    if os.name == 'nt':
        subprocess.call('cls', shell=True)
    else:
        subprocess.call('clear', shell=True)
