"""
Unit tests for ``Task``.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import joblib
import jsonpickle

from ecomapper.task.task import Task


class ConcreteTask(Task):
    def _get_input(self): pass

    def _run(self): pass

    def is_done(self): return True


class NotDoneConcreteTask(ConcreteTask):
    def is_done(self): return False


class AnotherTaskType(Task):
    def _get_input(self): pass

    def _run(self): pass

    def is_done(self): return True


class TestTask(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_basic_task(self):
        task = ConcreteTask(
            creation_date="2023-08-10",
            app_name="TestApp",
            app_version="1.0",
            root_dir=self.temp_dir.name,
            rng_seed=123,
            verbose=True
        )
        return task

    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    @patch("ecomapper.task.task.err")
    def test_task_creation(self, mock_err):
        # Ensure the temporary directory is empty
        self.assertEqual(len(os.listdir(self.temp_dir.name)), 0)

        # Create a task
        ConcreteTask(
            creation_date="2023-08-10",
            app_name="TestApp",
            app_version="1.0",
            root_dir=self.temp_dir.name,
            rng_seed=123,
            verbose=True
        )

        self.assertIn(Task.TASK_FILENAME, os.listdir(self.temp_dir.name))
        self.assertIn('journals', os.listdir(self.temp_dir.name))
        mock_err.assert_not_called()

    @patch("ecomapper.task.task.err")
    def test_task_creation_existing_dir(self, mock_err):
        # Create a file in the temporary directory to make it non-empty
        with open(os.path.join(self.temp_dir.name, 'testfile.txt'), 'w') as f:
            f.write('')

        with self.assertRaises(SystemExit):
            ConcreteTask(
                creation_date="2023-08-10",
                app_name="TestApp",
                app_version="1.0",
                root_dir=self.temp_dir.name,
                rng_seed=123,
                verbose=True)

        mock_err.assert_called_once()

    @patch("ecomapper.task.task.Task._try_save")
    def test_delattr_calls_try_save(self, mock_try_save):
        task = self.get_basic_task()

        # Ignore failed save attempts while Task is still loading
        mock_try_save.call_count = 0

        # Delete an attribute and ensure _try_save is called
        delattr(task, 'verbose')
        mock_try_save.assert_called_once()

    @patch("ecomapper.task.task.Task._try_save")
    def test_setattr_calls_try_save(self, mock_try_save):
        task = self.get_basic_task()

        # Ignore failed save attempts while Task is still loading
        mock_try_save.call_count = 0

        # Set an attribute and ensure _try_save is called
        task.new_attr = "test"
        mock_try_save.assert_called_once()

    def test_try_load_valid_task(self):
        task = self.get_basic_task()
        task._try_save()
        loaded_task = ConcreteTask.try_load(self.temp_dir.name)

        # Verify the task was correctly loaded
        self.assertIsNotNone(loaded_task)
        self.assertIsInstance(loaded_task, ConcreteTask)
        self.assertEqual(loaded_task.creation_date, "2023-08-10")
        self.assertEqual(loaded_task.app_name, "TestApp")
        self.assertEqual(loaded_task.app_version, "1.0")

    def test_try_load_incomplete_task(self):
        # Make a Task instance and delete a required attribute
        task = self.get_basic_task()
        delattr(task, 'creation_date')

        with open(os.path.join(self.temp_dir.name, Task.TASK_FILENAME),
                  'w') as f:
            f.write(jsonpickle.encode(task))

        with self.assertRaises(TypeError) as exc:
            ConcreteTask.try_load(self.temp_dir.name)
        self.assertTrue(
            "Task.__init__() missing 1 required positional argument: "
            "'creation_date'" in str(exc.exception))

    @patch("ecomapper.task.task.warn")
    def test_try_load_not_done_task(self, mock_warn):
        task = NotDoneConcreteTask(
            **self.get_basic_task().__dict__)

        with open(os.path.join(self.temp_dir.name, Task.TASK_FILENAME),
                  'w') as f:
            f.write(jsonpickle.encode(task))

        self.assertEqual(None, ConcreteTask.try_load(self.temp_dir.name))
        mock_warn.assert_called_once_with(
            f"Directory contains a valid {task.__class__.__name__} "
            f"but the task has not been completed yet")

    @patch("ecomapper.task.task.warn")
    def test_try_load_mismatched_task_type(self, mock_warn):
        task = AnotherTaskType(
            **self.get_basic_task().__dict__)

        with open(os.path.join(self.temp_dir.name, Task.TASK_FILENAME),
                  'w') as f:
            f.write(jsonpickle.encode(task))

        self.assertEqual(None, ConcreteTask.try_load(self.temp_dir.name))
        mock_warn.assert_called_once_with(
            f"Directory contains a {task.__class__.__name__}, "
            f"but a {ConcreteTask.__name__} is expected")

    @patch("ecomapper.utils.json_util.warn")
    def test_try_load_invalid_task_data(self, mock_warn):
        # Save invalid data to the file
        with open(os.path.join(self.temp_dir.name, Task.TASK_FILENAME),
                  'w') as f:
            f.write("invalid data")

        self.assertEqual(None, ConcreteTask.try_load(self.temp_dir.name))
        mock_warn.assert_called_once_with('Error parsing JSON file')

    @patch("ecomapper.task.task.warn")
    def test_try_load_from_invalid_path(self, mock_warn):
        self.assertEqual(None, Task.try_load('!'))
        mock_warn.assert_called_once_with("Path does not exist")

    @patch("ecomapper.task.task.warn")
    def test_try_load_from_directory_without_task(self, mock_warn):
        self.assertEqual(None, Task.try_load(self.temp_dir.name))
        mock_warn.assert_called_once_with("Directory does not contain a task")

    def test_to_string(self):
        task = self.get_basic_task()
        self.assertEqual(jsonpickle.encode(task, indent=4), str(task))
        self.assertEqual(jsonpickle.encode(task, indent=4), repr(task))

    @patch('os.path.exists', side_effect=lambda p: p == '/valid/path')
    @patch('ecomapper.task.task.warn')
    def test_some_invalid_ext_paths(self, mock_warn, _):
        task = self.get_basic_task()
        task.valid_path_ext = '/valid/path'
        task.invalid_path_ext = '/invalid/path'

        task._check_external_paths()

        # Assert valid path hasn't changed and
        # invalid path has been set to None
        self.assertEqual(task.valid_path_ext, '/valid/path')
        self.assertIsNone(task.invalid_path_ext)

        # Assert warn was called correctly
        mock_warn.assert_called_once_with(
            "Path to invalid_path_ext has become invalid: /invalid/path")

    def test_register_journal(self):
        task = self.get_basic_task()
        task.my_journal_file = task.register_journal('my_journal')

        self.assertTrue(hasattr(task, 'my_journal_file'))
        self.assertEqual(os.path.join(task.journal_dir, 'my_journal.joblib'),
                         task.my_journal_file)
        self.assertEqual(dict(), joblib.load(task.my_journal_file))

    @patch("tests.unit.task.test_task.ConcreteTask._check_external_paths")
    @patch("tests.unit.task.test_task.ConcreteTask._get_input")
    @patch("tests.unit.task.test_task.ConcreteTask._run")
    def test_run_task(self, mock_run, mock_input, mock_paths):
        task = self.get_basic_task()
        task.run()

        mock_paths.assert_called_once()
        mock_input.assert_called_once()
        mock_run.assert_called_once()
