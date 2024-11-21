"""
Functional tests for ``PipelineTask``.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from ecomapper.mmseg_interface.mmseg_config_wrapper import MMSegConfigWrapper
from ecomapper.mmseg_interface.registry import deregister_all
from tests.functional.utils.input_util import run_with_inputs
from tests.functional.utils.training_util import download_files_mock
from tests.functional.utils.training_util import dump_config
from tests.functional.utils.training_util import train_dataset_input


class TestPipelineTask(unittest.TestCase):
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

    def test_pipeline(self):
        os.environ["SCALE"] = "0.05"
        os.environ["SCALE_METHOD"] = "1"

        inputs = []
        if torch.cuda.device_count() > 0:
            inputs += [
                '1',  # num gpus
            ]

        inputs += ['2',  # batch size
                   '1',  # num epochs
                   '',  # confirm learning rate
                   '1',  # model choice = Mask2Former
                   '3'  # swin-s cityscapes
                   ]

        with tempfile.TemporaryDirectory() as temp_pipeline:
            run_with_inputs(
                'pipeline',
                temp_pipeline,
                ['n'] +
                train_dataset_input(256) + inputs + ['y',
                                                     os.path.join(
                                                         temp_pipeline,
                                                         'train_dataset')])

            del os.environ["SCALE"]
            del os.environ["SCALE_METHOD"]
