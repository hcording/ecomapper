"""
Print capture to assert what was printed during a test.
"""

import io
import sys


class PrintCapture(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        super().write(s)
