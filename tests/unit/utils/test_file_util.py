"""
Unit tests for ``file_util``.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecomapper.utils.file_util import atomic_action
from ecomapper.utils.file_util import atomic_write
from ecomapper.utils.file_util import file_to_list
from ecomapper.utils.file_util import get_unique_file_in_directory
from ecomapper.utils.file_util import list_to_file


class TestFileUtil(unittest.TestCase):

    def test_file_to_list(self):
        with tempfile.NamedTemporaryFile('w') as f:
            f.write("Hello\n")
            f.write("world")
            f.flush()
            result = file_to_list(f.name)
            self.assertListEqual(["Hello", "world"], result)

    def test_file_to_list_whitespace_ignored(self):
        with tempfile.NamedTemporaryFile('w') as f:
            f.write("Hello\n")
            f.write("  \n")
            f.write("world\n  ")
            f.write("  \n")
            f.flush()
            result = file_to_list(f.name)
            self.assertListEqual(["Hello", "world"], result)

    def test_list_to_file(self):
        with tempfile.NamedTemporaryFile('w+t') as f:
            inp = ["Hello", "world"]
            list_to_file(inp, f.name)
            self.assertEqual("Hello\nworld", f.read())

    def test_atomic_action_dir_path_raises(self):
        with self.assertRaises(AssertionError):
            with tempfile.TemporaryDirectory() as temp_dir:
                atomic_action([], temp_dir, lambda x, y: ())

    @staticmethod
    def placeholder_write(data, file):
        with open(file, 'w') as f:
            f.write(data)

    @patch('os.remove')
    @patch('shutil.move')
    @patch('tempfile.mkstemp')
    def test_atomic_action(self,
                           mock_mkstemp, mock_move, mock_remove):
        temp_file_1 = tempfile.NamedTemporaryFile(suffix='.txt')
        temp_file_2 = tempfile.NamedTemporaryFile(suffix='.txt')

        mock_mkstemp.return_value = (None, temp_file_2.name)
        mock_move.side_effect = lambda x, y: os.rename(x, y)

        atomic_action("Hello world",
                      temp_file_1.name, TestFileUtil.placeholder_write)

        mock_mkstemp.assert_called_once_with(dir=Path(temp_file_1.name).parent,
                                             suffix='.txt')
        mock_move.assert_called_once_with(temp_file_2.name, temp_file_1.name)
        mock_remove.assert_not_called()

        self.assertTrue(os.path.exists(temp_file_1.name))
        with open(temp_file_1.name, 'r') as f:
            self.assertEqual("Hello world", f.read())

        os.remove(temp_file_1.name)
        os.remove(temp_file_2.name)

    @patch('os.close')
    @patch('os.remove')
    @patch('shutil.move')
    @patch('tempfile.mkstemp')
    def test_atomic_action_move_interrupted(self, mock_mkstemp, mock_move,
                                            mock_remove, mock_close):
        temp_file_1 = tempfile.NamedTemporaryFile(suffix='.txt')
        temp_file_2 = tempfile.NamedTemporaryFile(suffix='.txt')

        fd = 1
        mock_mkstemp.return_value = (fd, temp_file_2.name)

        atomic_action("Hello world", temp_file_1.name,
                      TestFileUtil.placeholder_write)

        mock_mkstemp.assert_called_once_with(dir=Path(temp_file_1.name).parent,
                                             suffix='.txt')
        mock_move.assert_called_once_with(temp_file_2.name, temp_file_1.name)
        mock_remove.assert_called_once_with(temp_file_2.name)
        mock_close.assert_called_with(fd)

        self.assertTrue(os.path.exists(temp_file_1.name))
        with open(temp_file_1.name, 'r') as f:
            self.assertEqual("", f.read())

        os.remove(temp_file_1.name)
        os.remove(temp_file_2.name)

    def test_atomic_write(self):
        temp_file = tempfile.NamedTemporaryFile('w+t')
        n = temp_file.name
        atomic_write("Hello world", temp_file.name, 'w')

        with open(n, 'r') as f:
            self.assertEqual("Hello world", f.read())

        os.remove(n)

    def test_try_get_unique_file_in_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(dir=temp_dir) as f:
                result = get_unique_file_in_directory(temp_dir, '*')
                self.assertEqual(f.name, result)

    def test_get_unique_file_in_directory_with_two_files_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(dir=temp_dir):
                with tempfile.NamedTemporaryFile(dir=temp_dir):
                    with self.assertRaises(RuntimeError):
                        get_unique_file_in_directory(temp_dir, '*')

    def test_get_unique_file_in_directory_predicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(dir=temp_dir):
                with tempfile.NamedTemporaryFile(
                        dir=temp_dir, suffix='.txt') as f2:
                    result = get_unique_file_in_directory(
                        temp_dir, '*', lambda x: x.endswith('.txt'))
                    self.assertEqual(f2.name, result)
