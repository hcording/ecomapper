"""
Functional tests for ``PredictTask``.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from ecomapper.core.journal import Journal
from ecomapper.mmseg_interface.mmseg_config_wrapper import MMSegConfigWrapper
from ecomapper.mmseg_interface.registry import deregister_all
from ecomapper.task.predict_task import PredictTask
from tests.functional.utils.input_util import run_with_inputs
from tests.functional.utils.test_dataset_task_util import remove_half_journal
from tests.functional.utils.training_util import download_files_mock
from tests.functional.utils.training_util import dump_config
from tests.functional.utils.training_util import simple_train_input
from tests.functional.utils.training_util import train_dataset_input


class TestPredictTask(unittest.TestCase):

    def setUp(self) -> None:
        self.download_patcher = patch(
            'ecomapper.mmseg_interface.mmseg_config_wrapper'
            '.MMSegConfigWrapper._download_model_files',
            side_effect=download_files_mock).start()
        self.config_dump_patcher = patch.object(
            MMSegConfigWrapper,
            'dump_mmseg_config',
            side_effect=dump_config,
            autospec=True)
        self.config_dump_patcher.start()

    def tearDown(self) -> None:
        self.download_patcher.stop()
        self.config_dump_patcher.stop()
        deregister_all()

    def test_predict(self):
        os.environ["SCALE"] = "0.05"
        os.environ["SCALE_METHOD"] = "1"

        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_model:
                run_with_inputs('train', temp_model,
                                simple_train_input(temp_dataset))
                with tempfile.TemporaryDirectory() as temp_predictions:
                    run_with_inputs('predict', temp_predictions,
                                    [temp_model, temp_dataset])

        del os.environ["SCALE"]
        del os.environ["SCALE_METHOD"]

    def test_predict_after_predict_interruption(self):
        os.environ["SCALE"] = "0.05"
        os.environ["SCALE_METHOD"] = "1"

        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_model:
                run_with_inputs('train', temp_model,
                                simple_train_input(temp_dataset))
                with tempfile.TemporaryDirectory() as temp_predictions:
                    run_with_inputs('predict', temp_predictions,
                                    [temp_model, temp_dataset])
                    remove_half_journal(temp_predictions, PredictTask,
                                        'predict_journal_file')
                    run_with_inputs('predict', temp_predictions, [])

        del os.environ["SCALE"]
        del os.environ["SCALE_METHOD"]

    def test_predict_after_merge_interruption(self):
        os.environ["SCALE"] = "0.05"
        os.environ["SCALE_METHOD"] = "1"

        with tempfile.TemporaryDirectory() as temp_dataset:
            run_with_inputs('labeled-dataset',
                            temp_dataset, train_dataset_input(256))
            with tempfile.TemporaryDirectory() as temp_model:
                run_with_inputs('train', temp_model,
                                simple_train_input(temp_dataset))
                with tempfile.TemporaryDirectory() as temp_predictions:
                    run_with_inputs('predict', temp_predictions,
                                    [temp_model, temp_dataset])

                    predict_task = PredictTask.try_load(temp_predictions,
                                                        False)
                    self.assertIsNotNone(predict_task)
                    predict_task: PredictTask

                    os.remove(predict_task.segmentation_map_file)
                    merge_journal = Journal(predict_task.merge_journal_file)
                    progress = merge_journal['iter']
                    merge_journal['iter'] = \
                        progress[0] // 2, progress[1] // 2, progress[2] // 2

                    run_with_inputs('predict', temp_predictions, [])

        del os.environ["SCALE"]
        del os.environ["SCALE_METHOD"]
