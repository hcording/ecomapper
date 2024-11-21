"""
Abstract base class for MMSegmentation wrappers.
Children of this class have 3 responsibilities:

#. Implementing ``get_model_name``, which provides the generic display name of
   the wrapped model to show during model selection.

#. Implementing ``get_metafile``, which provides the path to the metafile.yaml
   file for the wrapped model in the mmsegmentation config directory.

#. Overriding ``__init__``. After calling ``super().__init__``,
   ``self.mmseg_config`` will be set. Child classes can then modify this
   config to make model-specific changes, if required.
"""

import importlib
import math
import os
import subprocess
from abc import ABC
from abc import abstractmethod
from importlib import metadata
from typing import TypeVar

from mmengine import Config

from ecomapper.utils.file_util import atomic_action
from ecomapper.utils.file_util import file_to_list
from ecomapper.utils.file_util import get_unique_file_in_directory
from ecomapper.utils.path_util import expand_path
from ecomapper.utils.print_util import success


class MMSegConfigWrapper(ABC):
    T = TypeVar('T', bound='MMSegConfigWrapper')

    @staticmethod
    @abstractmethod
    def get_model_name() -> str:
        """
        Name of the implemented model.
        Displayed as one of the options for which model to train
        with (e.g. Mask2Former or Segformer) when creating a TrainTask.

        :return: Model name.
        """
        ...  # pytype: disable=bad-return-type

    @staticmethod
    @abstractmethod
    def get_metafile() -> str:
        """
        Path to the metafile.yaml file in the MMSegmentation
        config directory for the model being wrapped by this class.

        :return: metafile.yaml filepath.
        """
        ...  # pytype: disable=bad-return-type

    @classmethod
    def get_model_variants(cls) -> list[dict]:
        """
        List of implemented model variants for the model being wrapped
        by ``cls``.

        :return: Variant list.
        """
        yaml = importlib.import_module('yaml')
        with open(os.path.join(MMSegConfigWrapper._locate_mmsegmentation(),
                               '.mim', 'configs',
                               cls.get_metafile()), 'rt') as f:
            return yaml.safe_load(f)['Models']

    def __init__(self, variant_config_file: str,
                 tile_width: int, tile_height: int, num_classes: int,
                 train_task):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.num_classes = num_classes

        env = os.environ.copy()
        if 'SCALE' in env:
            if 'SCALE_METHOD' not in env:
                raise RuntimeError(
                    "When setting the SCALE environment variable, "
                    "the SCALE_METHOD environment variable needs "
                    "to be set as well (to either 1 or 2)")
            try:
                scale = float(env['SCALE'])
            except ValueError as exc:
                raise RuntimeError("Given SCALE is not a number") from exc

            try:
                scale_method = int(env['SCALE_METHOD'])
            except ValueError as exc:
                raise RuntimeError(
                    "Given SCALE_METHOD is not an integer") from exc

            if scale_method == 1:
                self.tile_width = int(tile_width * scale)
                self.tile_height = int(tile_height * scale)

        num_train_samples = \
            len(file_to_list(train_task.train_split_file))

        self.mmseg_config = self._make_config(
            variant_config_file,
            train_task.model_working_dir,
            train_task.train_dataset_task_dir_ext,
            train_task.split_lists_dir,
            train_task.val_dataset_task_dir_ext
            if train_task.val_dataset_task_dir_ext else None,
            train_task.test_dataset_task_dir_ext
            if train_task.test_dataset_task_dir_ext else None,
            num_train_samples,
            train_task.device_batch_size,
            train_task.num_epochs,
            train_task.num_devices,
            train_task.learning_rate,
            train_task.rng_seed)

        self.dump_mmseg_config(self.mmseg_config.filename)

    def _make_config(self, mmseg_config_stub: str, model_working_dir: str,
                     train_dataset_task_dir_ext: str, split_lists_dir: str,
                     val_dataset_task_dir_ext: str | None,
                     test_dataset_task_dir_ext: str | None,
                     num_train_samples: int, device_batch_size: int,
                     num_epochs: int, num_devices: int,
                     learning_rate: float,
                     rng_seed: int) -> Config:
        """
        Downloads the config and backbone weights for the given
        ``mmseg_config_stub`` and sets default values that are needed for any
        model config.

        :param device_batch_size:
        :param num_epochs:
        :param num_devices:
        :param mmseg_config_stub:
        :param model_working_dir:
        :param split_lists_dir:
        :param train_dataset_task_dir_ext:
        :param val_dataset_task_dir_ext:
        :param test_dataset_task_dir_ext:
        :param num_train_samples:
        :param rng_seed:
        :return: Desired MMSegmentation config with default values.
        """
        # Download config and/or pretrained weights if either is missing.
        # The pretrained weights are not downloaded if another checkpoint
        # already exists.
        if (not MMSegConfigWrapper.try_get_mmseg_config_file(
                model_working_dir)
                or not MMSegConfigWrapper.try_get_latest_checkpoint(
                    model_working_dir)):
            MMSegConfigWrapper._download_model_files(model_working_dir,
                                                     mmseg_config_stub)
            success("Model files downloaded")

        config = Config.fromfile(
            MMSegConfigWrapper.try_get_mmseg_config_file(model_working_dir))

        # Modify dataset type and path
        config.dataset_type = 'CustomSegDataset'
        config.data_root = None

        # Set up working dir to save files and logs.
        config.work_dir = model_working_dir

        config.train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ScaleAugmentation'),
            dict(type='SelectiveAugmentation'),
            dict(type='PackSegInputs')
        ]

        config.test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='EvalScaleAugmentation'),
            dict(type='PackSegInputs')
        ]

        config.train_dataloader.dataset.type = config.dataset_type
        config.train_dataloader.dataset.data_root = None
        config.train_dataloader.num_workers = 2
        config.train_dataloader.persistent_workers = True
        config.train_dataloader.dataset.data_prefix = \
            dict(img_path=os.path.join(
                train_dataset_task_dir_ext,
                'image_tiles'),
                seg_map_path=os.path.join(
                    train_dataset_task_dir_ext,
                    'label_tiles'))
        config.train_dataloader.dataset.pipeline = config.train_pipeline
        config.train_dataloader.dataset.ann_file = \
            os.path.join(split_lists_dir, 'train.txt')

        # Use WeightedRandomSampler during training to mitigate class imbalance
        sampler = dict(
            type='WeightedInfiniteSampler',
            shuffle=True,
            seed=rng_seed,
            dataset=config.train_dataloader.dataset)

        config.train_dataloader.sampler = sampler

        MMSegConfigWrapper.configure_eval_dataloader(config,
                                                     config.val_dataloader,
                                                     val_dataset_task_dir_ext,
                                                     split_lists_dir,
                                                     'val.txt')
        config.val_dataloader.num_workers = 4
        config.val_dataloader.persistent_workers = True
        config.val_dataloader.sampler.shuffle = False
        config.val_evaluator = \
            dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

        MMSegConfigWrapper.configure_eval_dataloader(config,
                                                     config.test_dataloader,
                                                     test_dataset_task_dir_ext,
                                                     split_lists_dir,
                                                     'test.txt')
        config.test_dataloader.num_workers = 4
        config.test_dataloader.persistent_workers = True
        config.test_dataloader.sampler.shuffle = False
        config.test_evaluator = \
            dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

        config.train_dataloader.batch_size = device_batch_size
        config.val_dataloader.batch_size = device_batch_size
        config.test_dataloader.batch_size = device_batch_size

        config.optimizer.lr = learning_rate
        config.optim_wrapper.optimizer.lr = learning_rate

        config.load_from = \
            MMSegConfigWrapper.try_get_latest_checkpoint(model_working_dir)
        config.resume = False

        config.train_cfg.max_iters = \
            math.ceil(
                (num_train_samples * num_epochs) /
                (config.train_dataloader.batch_size * num_devices))

        # Validate at least every 500 steps
        # If the model is better, a checkpoint
        # will be created (best_mIoU_iter_X.pth)
        config.train_cfg.val_interval = \
            min(500, math.ceil(0.1 * config.train_cfg.max_iters))

        # Log statistics at least every 10 steps
        config.default_hooks.logger.interval = \
            min(10, math.ceil(0.1 * config.train_cfg.max_iters))

        # Also make at least 10 checkpoints for the training run
        config.default_hooks.checkpoint.interval = \
            math.ceil(0.1 * config.train_cfg.max_iters)

        # During inference, create a prediction at each iteration
        config.default_hooks.visualization.interval = 1

        config['randomness'] = dict(seed=rng_seed)

        vis_backend_save_dir = os.path.join(model_working_dir, 'vis_backend')
        config.vis_backends = [dict(type='LocalVisBackend',
                                    save_dir=vis_backend_save_dir)]

        config.visualizer = dict(
            type='SegLocalVisualizer', vis_backends=config.vis_backends,
            name='visualizer', save_dir=vis_backend_save_dir)

        # Initialize model with pretrained weights
        # map_location='cpu' reduces VRAM usage spike
        # see https://pytorch.org/docs/stable/generated/torch.load.html
        config.init_cfg = dict(
            type='Pretrained',
            checkpoint=MMSegConfigWrapper.try_get_init_weights_file(
                model_working_dir),
            map_location='cpu')

        return config

    @staticmethod
    def _download_model_files(model_working_dir: str,
                              mmseg_config_name: str) -> None:
        print("Downloading model files ...\n")

        subprocess.run(
            f"mim download mmsegmentation --config "
            f"\"{mmseg_config_name}\" "
            f"--dest \"{model_working_dir}\"",
            shell=True, check=True)

    @staticmethod
    def try_get_latest_checkpoint(model_working_dir: str) -> str | None:
        """
        Attempts to get the latest checkpoint path by reading the
        ``last_checkpoint`` file created in the ``model_working_dir`` during
        training.

        :param model_working_dir: Directory to search in.
        :return: The path to the latest checkpoint if it was found, otherwise
            the path to the pretrained weights, if they were downloaded.
        """
        last_checkpoint_file = get_unique_file_in_directory(
            model_working_dir, 'last_checkpoint')

        if last_checkpoint_file:
            last_checkpoint_file_lines = file_to_list(last_checkpoint_file)
            if len(last_checkpoint_file_lines) > 0:
                assert len(last_checkpoint_file_lines) == 1, \
                    ("last_checkpoint file is invalid, "
                     "should only contain 1 line")

                path = expand_path(last_checkpoint_file_lines[0])
                path = os.path.join(model_working_dir, os.path.basename(path))
                if path != "" and os.path.exists(path):
                    return path

        return MMSegConfigWrapper.try_get_init_weights_file(model_working_dir)

    @staticmethod
    def try_get_init_weights_file(model_working_dir: str) -> str | None:
        """
        Attempts to get the path to pretrained weights in the given
        ``model_working_dir``.

        :param model_working_dir: Directory to search in.
        :return: Path to the pretrained weights file if it was found, otherwise
            ``None``.
        """
        return get_unique_file_in_directory(
            model_working_dir, '*.pth', lambda x: 'iter' not in x)

    @staticmethod
    def try_get_mmseg_config_file(model_working_dir: str) -> str | None:
        """
        Attempts to get the MMSegmentation model configuration in the given
        ``model_working_dir``.

        :param model_working_dir: Directory to search in.
        :return: The path to the MMSegmentation model configuration file if it
            was found, otherwise ``None``.
        """
        return get_unique_file_in_directory(model_working_dir, '*.py')

    @staticmethod
    def configure_eval_dataloader(mmseg_config, data_loader, dataset_task_dir,
                                  split_lists_dir, split_file):
        """
        Configures an evaluation (validation or testing) dataloader.

        :param mmseg_config: The MMSegmentation config owning the dataloader.
        :param data_loader: The dataloader to configure.
        :param dataset_task_dir: Optional ``DatasetTask`` for validation or
            testing. Only given if separate datasets are used for training,
            validation, and testing.
        :param split_lists_dir: See ``TrainTask.split_lists_dir``.
        :param split_file: See ``TrainTask.val_split_file`` and
            ``TrainTask.test_split_file``.
        :return: Configured dataloader.
        """
        dataset = data_loader.dataset
        dataset.type = mmseg_config.dataset_type
        dataset.data_root = None

        # Images are sourced from the train_dataloader if there are no
        # separate datasets for validation and testing
        dataset.data_prefix = (
            mmseg_config.train_dataloader.dataset.data_prefix
            if not dataset_task_dir else
            dict(img_path=os.path.join(
                dataset_task_dir,
                'image_tiles'),
                seg_map_path=os.path.join(
                    dataset_task_dir,
                    'label_tiles')))

        dataset.pipeline = mmseg_config.test_pipeline
        dataset.ann_file = os.path.join(split_lists_dir, split_file)

    @staticmethod
    def _locate_mmsegmentation() -> str:
        """
        Helper method for finding the MMSegmentation sources.

        :return: Path to MMSegmentation source directory.
        """
        distribution = metadata.distribution('mmsegmentation')
        mmsegmentation_dir = str(distribution.locate_file('mmseg'))
        return mmsegmentation_dir

    def dump_mmseg_config(self, mmseg_config_file: str):
        atomic_action(
            self.mmseg_config,
            mmseg_config_file,
            lambda x, y: x.dump(y))
