"""
Context manager to gracefully exit from signals and exceptions during critical
code segments.
"""

import os
import signal
from types import FrameType
from typing import Any
from typing import Callable
from typing import Type

from ecomapper.utils.signal_util import ignore_all_signals
from ecomapper.utils.signal_util import restore_all_signals
from ecomapper.utils.signal_util import set_handler_for_all_signals
from ecomapper.utils.signal_util import stop_signals


class SafeContextManager:
    def __init__(self,
                 setup: Callable[[], Any] | None = None,
                 cleanup: Callable[[signal.Signals | Type[
                     BaseException] | None], Any] | None = None):
        self.setup = setup
        self.cleanup = cleanup
        self.raised_signal: signal.Signals | Type[BaseException] | None = None
        self.cleanup_done = False

    def handle_signal(self,
                      sig: signal.Signals | Type[BaseException] | None,
                      stack_trace: FrameType | None = None) -> None:
        """
        Handles an incoming signal while inside the manager's scope.
        The following procedure is executed:

        #. If the current PID does match the PID of the process owning this
           ``SafeContextManger`` instance, return.

        #. If the raised signal was previously handled or is being handled
           lower in the call stack, return.

        #. Ignore all future signals

        #. Call ``self.cleanup``.

        #. Restore default behavior for all signals.

        #. Reraise the received signal.

        :param sig: Incoming signal.
        :param stack_trace: Stack trace.
        """

        # Only handle signals on the process that entered the context
        # Otherwise signals may get handled more than once
        if os.getpid() != self.enter_pid:
            return

        # Don't handle the signal if it was already handled or is
        # being handled somewhere lower in the call stack
        if self.raised_signal is not None:
            return

        self.raised_signal = sig

        ignore_all_signals()

        # Cleanup without interruptions (except SIGKILL)
        if self.cleanup is not None and not self.cleanup_done:
            self.cleanup(sig)
            self.cleanup_done = True

        restore_all_signals()

        if self.raised_signal is not None:
            if self.raised_signal in stop_signals:
                # Interpret all interrupts as SIGINT to get
                # a ``KeyboardInterrupt`` instead of an abrupt termination
                signal.raise_signal(signal.SIGINT)
            else:
                # All other signals are reraised normally
                signal.raise_signal(self.raised_signal)

    def __enter__(self) -> None:
        """
        Enters the context, calling ``self.setup`` and registering
        ``self.handle_signal`` as handler for all signals.
        """

        # Register the owner of this context
        self.enter_pid = os.getpid()

        set_handler_for_all_signals(lambda x, y: self.handle_signal(x, y))

        # Setup can be interrupted, in which case cleanup will be called
        if self.setup is not None:
            self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context, calling ``self.cleanup`` and restoring
        the default signal handlers.
        """

        # Only cleanup if no signal or exception occurred
        if self.cleanup is not None and not self.cleanup_done:
            self.handle_signal(exc_type, None)
            self.cleanup_done = True
