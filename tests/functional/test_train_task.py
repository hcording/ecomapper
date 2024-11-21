"""
Functional tests for ``TrainTask``.
"""

import os.path
import os.path
import sys
import tempfile
import unittest
from glob import glob
from unittest.mock import patch

from mmengine import Config

from ecomapper.mmseg_interface.registry import deregister_all
from ecomapper.task.train_task import TrainTask
from tests.functional.utils.input_util import run_with_inputs
from tests.functional.utils.print_capture import PrintCapture
from tests.functional.utils.training_util import download_files_mock
from tests.functional.utils.training_util import separate_train_input
from tests.functional.utils.training_util import simple_train_input
from tests.functional.utils.training_util import train_dataset_input


class TestTrainTask(unittest.TestCase):

    @staticmethod
    def train_mock(task: TrainTask, mmseg_config: Config):
        checkpoints = [os.path.join(task.model_working_dir, f'iter_{i}.pth')
                       for i in range(1, 6)]
        for c in checkpoints:
            with open(c, 'w'):
                pass

        with open(os.path.join(task.model_working_dir, 'last_checkpoint'),
                  'w') as f:
            f.write(checkpoints[-1])

        deregister_all()

    @staticmethod
    def eval_mock(task: TrainTask, mmseg_config_file: str):
        task.eval_done = True
        deregister_all()

    def setUp(self) -> None:
        self.mmseg_train_patcher = patch(
            'ecomapper.task.train_task._train',
            side_effect=TestTrainTask.train_mock).start()
        self.mmseg_eval_patcher = patch(
            'ecomapper.task.train_task._evaluate',
            side_effect=TestTrainTask.eval_mock).start()

        self.patcher = patch(
            'ecomapper.mmseg_interface.mmseg_config_wrapper'
            '.MMSegConfigWrapper._download_model_files',
            side_effect=download_files_mock).start()

    def tearDown(self) -> None:
        self.patcher.stop()
        self.mmseg_train_patcher.stop()
        self.mmseg_eval_patcher.stop()
        deregister_all()

    def test_new_train(self):
        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_dir:
                run_with_inputs('train', temp_dir,
                                simple_train_input(temp_dataset))

    def test_new_train_scale_method_1(self):
        os.environ["SCALE"] = "0.1"
        os.environ["SCALE_METHOD"] = "1"
        stdout_capture = PrintCapture()
        sys.stdout = stdout_capture

        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_dir:
                run_with_inputs('train', temp_dir,
                                simple_train_input(temp_dataset))

        del os.environ["SCALE"]
        del os.environ["SCALE_METHOD"]

        sys.stdout.seek(0)
        output = sys.stdout.getvalue()  # type: ignore
        sys.stdout = sys.__stdout__

        self.assertTrue("Found SCALE environment variable" in output)

    def test_new_train_scale_method_2(self):
        os.environ["SCALE"] = "0.1"
        os.environ["SCALE_METHOD"] = "2"
        stdout_capture = PrintCapture()
        sys.stdout = stdout_capture

        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_dir:
                run_with_inputs('train', temp_dir,
                                simple_train_input(temp_dataset))

        del os.environ["SCALE"]
        del os.environ["SCALE_METHOD"]

        sys.stdout.seek(0)
        output = sys.stdout.getvalue()  # type: ignore
        sys.stdout = sys.__stdout__

        self.assertTrue("Found SCALE environment variable" in output)

    def test_new_train_scale_method_without_scale_raises(self):
        os.environ["SCALE_METHOD"] = "1"

        with self.assertRaises(RuntimeError):
            with tempfile.TemporaryDirectory() as temp_dataset:
                run_with_inputs('labeled-dataset',
                                temp_dataset, train_dataset_input(256))
                with tempfile.TemporaryDirectory() as temp_dir:
                    run_with_inputs('train', temp_dir,
                                    simple_train_input(temp_dataset))

        del os.environ["SCALE_METHOD"]

    def test_new_train_scale_without_scale_method_raises(self):
        os.environ["SCALE"] = "0.5"

        with self.assertRaises(RuntimeError):
            with tempfile.TemporaryDirectory() as temp_dataset:
                run_with_inputs('labeled-dataset',
                                temp_dataset, train_dataset_input(256))
                with tempfile.TemporaryDirectory() as temp_dir:
                    run_with_inputs('train', temp_dir,
                                    simple_train_input(temp_dataset))

        del os.environ["SCALE"]

    @staticmethod
    def test_existing_train():
        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_dir:
                run_with_inputs('train', temp_dir,
                                simple_train_input(temp_dataset))
                run_with_inputs('train', temp_dir, [])

    @staticmethod
    def test_new_train_separate_val_test_datasets():
        with tempfile.TemporaryDirectory() as train_dataset:
            run_with_inputs('labeled-dataset',
                            train_dataset, train_dataset_input(512))
            with tempfile.TemporaryDirectory() as val_dataset:
                run_with_inputs('labeled-dataset',
                                val_dataset, train_dataset_input(512))
                with tempfile.TemporaryDirectory() as test_dataset:
                    run_with_inputs('labeled-dataset',
                                    test_dataset, train_dataset_input(512))
                    with tempfile.TemporaryDirectory() as temp_dir:
                        run_with_inputs('train', temp_dir,
                                        separate_train_input(
                                            train_dataset, val_dataset,
                                            test_dataset))

    @staticmethod
    def test_existing_separate_val_test_datasets():
        with tempfile.TemporaryDirectory() as train_dataset:
            run_with_inputs('labeled-dataset',
                            train_dataset, train_dataset_input(512))
            with tempfile.TemporaryDirectory() as val_dataset:
                run_with_inputs('labeled-dataset',
                                val_dataset, train_dataset_input(512))
                with tempfile.TemporaryDirectory() as test_dataset:
                    run_with_inputs('labeled-dataset',
                                    test_dataset, train_dataset_input(512))
                    with tempfile.TemporaryDirectory() as temp_dir:
                        run_with_inputs('train', temp_dir,
                                        separate_train_input(
                                            train_dataset, val_dataset,
                                            test_dataset))
                        run_with_inputs('train', temp_dir, [])

    def test_continue_after_interrupt_train(self):
        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_dir:
                run_with_inputs('train', temp_dir,
                                simple_train_input(temp_dataset))
                task = TrainTask.try_load(temp_dir, require_done=False)
                self.assertIsNotNone(task)
                task: TrainTask

                checkpoints = sorted(glob(os.path.join(
                    task.model_working_dir, '*.pth')))
                task.last_checkpoint_file = checkpoints[len(checkpoints) // 2]
                for cp in checkpoints[len(checkpoints) // 2 + 1:]:
                    os.remove(cp)
                with open(os.path.join(task.model_working_dir,
                                       'last_checkpoint'), 'w') as f:
                    f.write(checkpoints[len(checkpoints) // 2])

                run_with_inputs('train', temp_dir, [])
