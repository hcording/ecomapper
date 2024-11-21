"""
Utility for testing ``TrainTask`` functions.
"""

import math
import os
import shutil

import torch.cuda

from ecomapper.mmseg_interface.mmseg_config_wrapper import MMSegConfigWrapper
from ecomapper.utils.file_util import atomic_action
from tests.functional.utils.input_util import _here


def train_input_image_file():
    return os.path.join(train_input_dir(), 'image.tif')


def train_input_label_map_file():
    return os.path.join(train_input_dir(), 'label_map.tif')


def train_input_label_legend_file():
    return os.path.join(train_input_dir(), 'label_legend.json')


def train_input_dir():
    return os.path.join(_here().parent.parent, 'data', 'mock_train_inputs')


def train_dataset_input(tile_side_length: int):
    return [
        str(tile_side_length),  # tile width
        str(tile_side_length),  # tile height
        train_input_image_file(),  # input image?
        train_input_label_map_file(),  # label map?
        'chayote', 'border', ''  # label legend
    ]


def simple_train_input(dataset_directory: str):
    inputs = [
        dataset_directory,  # dataset directory?
        'no',  # separate val and test datasets?
    ]

    if torch.cuda.device_count() > 0:
        inputs += [
            '1'  # num gpus
        ]

    inputs += ['2',  # batch size
               '1',  # num epochs
               '',  # confirm learning rate
               '1',  # model choice = Mask2Former
               '3'  # swin-s cityscapes
               ]

    return inputs


def separate_train_input(train_dataset: str,
                         val_dataset: str,
                         test_dataset: str):
    inputs = [
        train_dataset,  # dataset directory?
        'yes',  # separate val and test datasets?
        val_dataset,  # validation dataset?
        test_dataset,  # test dataset?
    ]

    if torch.cuda.device_count() > 0:
        inputs += [
            '1'  # num gpus
        ]

    inputs += [
        '2',  # batch size
        '1',  # num epochs
        '',  # confirm learning rate
        '1',  # model choice = Mask2Former
        '3'  # swin-s cityscapes
    ]

    return inputs


def download_files_mock(
        model_working_dir, mmseg_config_name):
    mmseg_config_file = os.path.join(
        model_working_dir,
        mmseg_config_name + '.py')

    shutil.copy(os.path.join(
        train_input_dir(),
        mmseg_config_name + '.py'),
        mmseg_config_file)


def dump_config(wrapper: MMSegConfigWrapper, mmseg_config_file: str):
    wrapper.mmseg_config.train_cfg.val_interval = \
        math.ceil(0.9 * wrapper.mmseg_config.train_cfg.max_iters)

    wrapper.mmseg_config.default_hooks.logger.interval = 1

    wrapper.mmseg_config.default_hooks.checkpoint.interval = \
        math.ceil(0.9 * wrapper.mmseg_config.train_cfg.max_iters)

    wrapper.mmseg_config.vis_backends = []
    wrapper.mmseg_config.visualizer = None

    if 'visualization' in wrapper.mmseg_config.default_hooks:
        del wrapper.mmseg_config.default_hooks['visualization']

    atomic_action(
        wrapper.mmseg_config,
        mmseg_config_file,
        lambda x, y: x.dump(y))
