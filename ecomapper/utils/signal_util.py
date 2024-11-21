"""
Custom signal handlers.
"""

import signal
from types import FrameType
from typing import Callable

stop_signals = [
    signal.SIGHUP, signal.SIGINT,
    signal.SIGQUIT, signal.SIGABRT,
    signal.SIGTERM, signal.SIGSTOP]
"""
Signals that stop or suspend the application.
"""


def restore_all_signals() -> None:
    """
    Restores the default handler for all signals.
    """
    for sig in signal.valid_signals():
        try:
            signal.signal(sig, signal.SIG_DFL)
        except OSError:
            pass


def set_handler_for_all_signals(
        handler: Callable[[int | signal.Signals,
                           FrameType | None], None]) -> None:
    """
    Sets the given handler for all signals.

    :param handler: Signal handler.
    """
    for sig in signal.valid_signals():
        try:
            signal.signal(sig, handler)
        except OSError:
            pass


def ignore_all_signals() -> None:
    """
    Sets the handler for all signals to ``signal.SIG_IGN``, thereby ignoring
    them.
    """
    for sig in signal.valid_signals():
        try:
            signal.signal(sig, signal.SIG_IGN)
        except OSError:
            pass
