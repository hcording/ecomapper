"""
Mocking of input to simulate user input during tests.
"""

import sys
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from ecomapper.main import main as run_app


def _here():
    return Path(__file__).parent.expanduser().resolve()


def run_with_inputs(task: str, directory: str, inputs: list[str],
                    other_args: list | None = None):
    with (MonkeyPatch().context() as monkeypatch):
        inputs = iter(inputs)
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))
        sys.argv = ([sys.argv[0], task, directory]
                    + (other_args if other_args is not None else []))
        run_app()
