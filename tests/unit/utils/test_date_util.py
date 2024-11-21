"""
Unit tests for ``date_util``.
"""

import datetime
import unittest
from unittest import mock

from ecomapper.utils import date_util
from ecomapper.utils.date_util import convert_to_humanly_readable_date
from ecomapper.utils.date_util import filename_friendly_date


class TestDateUtil(unittest.TestCase):

    @mock.patch(f'{date_util.__name__}.datetime', wraps=datetime)
    def test_filename_friendly_date(self, mock_datetime):
        now = datetime.datetime(2023, 1, 20, 3, 44, 16)
        mock_datetime.datetime.now.return_value = now
        expected = "Jan_20_2023-03_44_16"
        actual = filename_friendly_date()
        self.assertEqual(expected, actual)

    @mock.patch(f'{date_util.__name__}.datetime', wraps=datetime)
    def test_convert_to_humanly_readable_date(self, mock_datetime):
        now = datetime.datetime(2023, 1, 20, 3, 44, 16)
        mock_datetime.datetime.now.return_value = now
        expected = "20 Jan 2023 at 03:44:16"
        actual = convert_to_humanly_readable_date(filename_friendly_date())
        self.assertEqual(expected, actual)
