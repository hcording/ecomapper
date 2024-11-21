"""
Prompts, i.e., reading input from the user, automatic retry, undo, and
validity checking.
"""

import importlib
import os
from typing import Callable, TypeVar

from ecomapper.utils.path_util import expand_path
from ecomapper.utils.print_util import CommonMessages, warn

ASSUME_DEFAULT = False


def pad_prompt(prompt: str) -> str:
    """
    Appends a space as the last character to the given prompt, if no space
    is present yet.

    :param prompt: Prompt to pad.
    :return: Padded prompt.
    """
    if len(prompt) > 0 and prompt[-1] != ' ':
        prompt += ' '
    return prompt


T = TypeVar('T')


def prompt_with_predicates(prompt: str,
                           predicates: list[tuple[Callable[[str], T], str]]
                           ) -> tuple[str, list[T]]:
    """
    Prompts the user for input which must satisfy the given predicate(s).

    :param prompt: Prompt to print before caret.
    :param predicates: List of tuples (predicate, failure message). A predicate
        can return any value that is interpretable as truthy or falsy.
    :return: User input satisfying all predicates, and results collected
        from each predicate.

    Notes
    -----
    The results returned by all predicates are assumed to have the same type.
    While this is not strictly enforced, violating this assumption will break
    type hints.
    """
    prompt = pad_prompt(prompt)
    result = input(prompt)
    predicate_results = []

    success = False
    while not success:
        success = True
        predicate_results = []
        for predicate, message in predicates:
            predicate_result = predicate(result)
            success = success and predicate_result
            if not predicate_result:
                if message != '':
                    warn(message)
                break
            predicate_results.append(predicate_result)
        if success:
            break
        result = input(prompt)
    return result, predicate_results

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def prompt_integer(
        prompt: str, low: int | None = None,
        high: int | None = None,
        default: int | None = None,
        predicates: list[tuple[
            Callable[[str], bool], str]] | None = None,
        hide_default: bool = False) -> int:
    """
    Prompts the user for an integer in the range [low, high].

    :param prompt: Prompt to print before caret.
    :param low: Lower bound (inclusive).
    :param high: Upper bound (inclusive).
    :param default: Choice to return if user pressed enter
        without providing any other input.
    :param predicates: List of tuples (predicate, failure message).
    :param hide_default: If ``True``, the default option is not shown in
        brackets as part of the prompt.
    :return: Integer in given range.
    """
    predicates = [] if predicates is None else predicates
    return prompt_numeric_value(  # pytype: disable=bad-return-type
        prompt, low, high, default,
        predicates + [
            (lambda x: (default is not None and x == '')
                     or (_is_number(x) and float(x).is_integer()),
             "Expecting an integer")
        ],
        hide_default
    )

def prompt_numeric_value(
        prompt: str, low: float | int | None = None,
        high: float | int | None = None,
        default: float | int | None = None,
        predicates: list[tuple[
            Callable[[str], bool], str]] | None = None,
        hide_default: bool = False) -> float | int:
    """
    Prompts the user for a number in the range [low, high].

    :param prompt: Prompt to print before caret.
    :param low: Lower bound (inclusive).
    :param high: Upper bound (inclusive).
    :param default: Choice to return if user pressed enter
        without providing any other input.
    :param predicates: List of tuples (predicate, failure message).
    :param hide_default: If ``True``, the default option is not shown in
        brackets as part of the prompt.
    :return: Number in given range.
    """
    if predicates is None:
        predicates = []

    if default and not hide_default:
        if prompt.endswith(':'):
            prompt = prompt[:-1]
        if not prompt.endswith(' '):
            prompt += ' '
        prompt += f'[{default}]:'

    result = prompt_with_predicates(
        prompt,
        # ENTER returns ``default`` or simply repeats the prompt
        [(lambda x: (default is not None and x == '') or len(x) > 0, ''),

         # ``default`` set and user pressed ENTER, or input must be numeric
         (lambda x: (default is not None and x == ''
                     or _is_number(x)),
          "Input must be a number"),

         # ``default`` set and user pressed ENTER, or input must be in range
         (lambda x: True if (x == '' and default is not None) else (
                 (float(x) - 1 if low is None else low)
                 <= float(x) <= (float(x) + 1 if high is None else high)),
          "Input must be in valid range: "
          f"{'(-inf, ' if low is None else '[' + str(low) + ', '}"
          f"{'inf)' if high is None else str(high) + ']'}")]
        + predicates)[0]  # append any user predicates and get prompt result

    if default and result == '':
        return default

    # Must now be parsable
    result = float(result)
    return int(result) if result.is_integer() else result


def prompt_image(prompt: str) -> str:
    """
    Prompts the user for a path, which must point to an image that can
    be opened with GDAL. All common image formats are supported.

    :param prompt: Prompt to print before caret.
    :return: Provided path.
    """
    try_load_image = importlib.import_module(
        'ecomapper.utils.geo_image_util').try_load_image

    return expand_path(prompt_with_predicates(
        prompt,
        [(lambda x: os.path.isfile and (
                try_load_image(x) is not None), "")])[0])


def prompt_yes_no(prompt: str, default: bool) -> bool:
    """
    Prompts a yes/no question.

    :param prompt: Question to ask.
    :param default: Default choice to make when pressing return.
    :return: Boolean indicating the user's choice.
    """
    prompt = pad_prompt(prompt)

    prompt += f"({'Y' if default is not None and default else 'y'}/" \
              f"{'N' if default is not None and not default else 'n'}) "

    while True:
        if ASSUME_DEFAULT:
            choice = ''
        else:
            choice = input(prompt).strip().lower()

        if choice == 'y' or choice == 'yes' or (
                default is not None and default and choice == ''):
            return True
        if choice == 'n' or choice == 'no' or (
                default is not None and not default and choice == ''):
            return False

        warn(f"{CommonMessages.INVALID_INPUT_RETRY}")
