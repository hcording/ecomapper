"""
Model training and evaluation.
"""

import os
import subprocess
import sys
from pathlib import Path
from random import shuffle
from time import sleep
from typing import Any

import torch
from mmengine import Config
from mmengine.runner import Runner

from ecomapper.core.journal import Journal
from ecomapper.launchers.mmseg_test import main as mmseg_test
from ecomapper.mmseg_interface.mmseg_config_wrapper import MMSegConfigWrapper
from ecomapper.mmseg_interface.registry import deregister_all, \
    register_eval_components, register_train_components
from ecomapper.task.dataset_task import DatasetTask
from ecomapper.task.task import Task
from ecomapper.task.unlabeled_dataset_task import UnlabeledDatasetTask
from ecomapper.utils.file_util import get_unique_file_in_directory, \
    list_to_file
from ecomapper.utils.joblib_util import joblib_dump_atomic, joblib_load_typed
from ecomapper.utils.path_util import get_stems, make_dir
from ecomapper.utils.print_util import warn
from ecomapper.utils.prompt_util import prompt_numeric_value, \
    prompt_with_predicates, prompt_yes_no, prompt_integer
from ecomapper.utils.random_util import set_torch_seed
from ecomapper.utils.torch_util import is_running_with_multiple_gpus


class TrainTask(Task):

    @property
    def default_device_batch_size(self):
        """
        Default *per-device* batch size (number of samples to compute a loss on
        and update model weights with) for training.

        :return: Default device batch size.
        """
        return 2

    @property
    def default_num_epochs(self):
        """
        Default number of epochs (iterations over the entire training set).

        :return: Default number of epochs.
        """
        return 50

    @property
    def default_num_devices(self):
        """
        Default number of devices to use.

        :return: Number of visible GPUs, or 1 if none are available.
        """
        return max(1, torch.cuda.device_count())

    @property
    def default_learning_rate(self):
        """
        Default learning rate.

        :return: Default learning rate.
        """
        return 0.0001

    def __init__(self,
                 train_dataset_task_dir_ext: str | None = None,
                 separate_datasets: bool = None,
                 val_dataset_task_dir_ext: str | None = None,
                 test_dataset_task_dir_ext: str | None = None,
                 device_batch_size: int | None = None,
                 num_epochs: int | None = None,
                 learning_rate: float | None = None,
                 num_devices: int | None = None,
                 model_choice: MMSegConfigWrapper | None = None,
                 model_variant: str | None = None,
                 eval_done: bool = False,
                 **kwargs: dict[str, Any]):
        super().__init__(**kwargs)

        self.SCALE: float | None = kwargs.get("SCALE", None)  # pytype: disable=annotation-type-mismatch
        """
        Scale factor between 0 and 1 that can be configured by setting the
        ``SCALE`` environment variable. Tiles will be scaled by this amount,
        using the method specified by ``self.SCALE_METHOD``.
        """

        self.SCALE_METHOD: int | None = kwargs.get("SCALE_METHOD", None)  # pytype: disable=annotation-type-mismatch
        """
        Method for scaling tiles that can be configured by setting the
        ``SCALE_METHOD`` environment variable. A value of 1 means downscaling
        both the image and label tiles by ``self.SCALE``. A value of 2 means
        downscaling image tiles by ``self.SCALE`` and upscaling them back to
        their original size. In method 2, label tiles are unaffected.
        """

        if not self.SCALE or not self.SCALE_METHOD:
            self.SCALE = None
            self.SCALE_METHOD = None

            scale, scale_method = TrainTask._get_scale_env_vars()
            if scale:
                self.SCALE = scale
                self.SCALE_METHOD = scale_method

        self.train_dataset_task_dir_ext = train_dataset_task_dir_ext
        """
        ``DatasetTask`` to use for training. If ``self.separate_datasets`` is
        ``False``, this dataset will be split into train, validation, and test
        sets.
        """

        self.separate_datasets = separate_datasets
        """
        Whether to use separate datasets for training, validation, and testing.
        """

        self.val_dataset_task_dir_ext = val_dataset_task_dir_ext
        """
        Optional path to a ``DatasetTask`` to use for validation.
        """

        self.test_dataset_task_dir_ext = test_dataset_task_dir_ext
        """
        Optional path to a ``DatasetTask`` to use for testing.
        """

        self.device_batch_size = device_batch_size
        """
        Number of samples to process *per-device* in a single training
        iteration. The effective batch size is ``self.device_batch_size *
        self.num_devices``.
        """

        self.num_epochs = num_epochs
        """
        Number of epochs to train for. An epoch is a single pass over the
        entire training set.
        """

        self.num_devices = num_devices
        """
        Number of devices in the system to use for training.
        This will be 1 if the system has no GPUs.
        """

        self.learning_rate = learning_rate
        """
        Learning rate for model training.
        """

        self.split_lists_dir = make_dir(
            os.path.join(self.root_dir, 'split_lists'))
        """
        Path to directory where the train/validation/test splits are stored as
        textfiles.
        """

        self.train_split_file = \
            os.path.join(self.split_lists_dir, 'train.txt')
        """
        Path to textfile storing filenames (without extension) to use for
        training, separated by newlines.
        For example, if the training set contains an image "tile_01.jpg"
        and a corresponding mask "tile_01.png", then train.txt would contain
        "tile_01".
        """

        self.val_split_file = \
            os.path.join(self.split_lists_dir, 'val.txt')
        """
        Path to textfile storing filenames (without extension) to use for
        validation, separated by newlines.
        For example, if the validation set contains an image "tile_01.jpg"
        and a corresponding mask "tile_01.png", then val.txt would contain
        "tile_01".
        """

        self.test_split_file = \
            os.path.join(self.split_lists_dir, 'test.txt')
        """
        Path to textfile storing filenames (without extension) to use for
        testing, separated by newlines.
        For example, if the testing set contains an image "tile_01.jpg"
        and a corresponding mask "tile_01.png", then test.txt would contain
        "tile_01".
        """

        self.sample_weights_file = \
            os.path.join(self.root_dir, 'sample_weights.joblib')
        """
        Path to ``joblib``-compressed dictionary mapping filenames (without
        extension) to their weight. The higher the weight, the higher the
        probability of a file being included in the next batch for training.
        """

        self.model_working_dir = make_dir(
            os.path.join(self.root_dir, 'model_working_dir'))
        """
        Path to working directory of the MMSegmentation model.
        Contains the MMSegmentation config, checkpoint files,
        and statistics about training, validation, and testing.
        """

        self.mmseg_config_wrapper_file = \
            os.path.join(self.model_working_dir, 'mmseg_config_wrapper.joblib')
        """
        Path to ``joblib``-compressed ``MMSegConfigWrapper`` instance,
        which details config modifications that have to be made
        for the chosen MMSegmentation model.
        """

        self.model_choice = model_choice
        """
        Model to train with, chosen by user.
        """

        self.model_variant = model_variant
        """
        Specifies the exact variant of the chosen model.
        This dictates the model's backbone, which dataset it was pretrained on,
        model size, etc.
        """

        self.eval_done = eval_done
        """
        ``True`` if the model has been evaluated, ``False`` otherwise.
        """

        set_torch_seed(self.rng_seed)

    @staticmethod
    def _get_scale_env_vars() -> tuple[float, int] | tuple[None, None]:
        """
        Checks for the ``SCALE`` and ``SCALE_METHOD`` environment variables and
        returns them if they are present.

        :return: A ``tuple`` of ``(scale, scale_method)``, or a ``tuple`` of
            ``(None, None)`` if the environment variables were not specified.
        :raise RuntimeError: If SCALE or SCALE_METHOD environment variable have
            invalid values, or only one of them is present.
        """
        env = os.environ.copy()
        if 'SCALE' in env:
            if 'SCALE_METHOD' not in env:
                raise RuntimeError(
                    "When setting the SCALE environment variable, "
                    "the SCALE_METHOD environment variable needs "
                    "to be set as well")
            try:
                scale = float(env['SCALE'])
                if scale < 0 or scale > 1:
                    raise RuntimeError(
                        "Given scale value is invalid, "
                        "must be greater than 0 and less than 1")
            except ValueError as exc:
                raise RuntimeError(
                    "Given scale value is not a number") from exc

            try:
                scale_method = int(env['SCALE_METHOD'])
            except ValueError as exc:
                raise RuntimeError(
                    "Given scale method is not an integer") from exc

            if not 0 < scale_method < 3:
                raise RuntimeError(f"Invalid scale method: {scale_method} "
                                   f"(must be 1 or 2)")

            print(f"Found SCALE environment variable, "
                  f"tiles will be scaled by a factor of {scale}")

            return scale, scale_method

        if 'SCALE_METHOD' in env:
            if 'SCALE' not in env:
                raise RuntimeError(
                    "When setting the SCALE_METHOD environment variable, "
                    "the SCALE environment variable needs "
                    "to be set as well")

        return None, None

    def _get_input(self) -> None:
        train_done = self.is_training_done()
        eval_done = self.is_evaluation_done()

        train_dataset_task, val_dataset_task, test_dataset_task = \
            self._get_datasets(eval_done, train_done)

        if not self.train_test_split_done():
            print("Calculating train/val/test split ...")
            self._make_train_val_test_split(
                train_dataset_task,
                val_dataset_task,
                test_dataset_task,
                (0.8, 0.2),
                self.separate_datasets)

        self.num_devices = self._get_num_devices()
        self.device_batch_size = self._get_device_batch_size()
        self.num_epochs = self._get_num_epochs()
        self.learning_rate = self._get_learning_rate()

        self.mmseg_config_wrapper_file = os.path.join(
            self.model_working_dir,
            'mmseg_config_wrapper.joblib')

        if (self.model_choice is None
                or self.model_variant is None):
            self.model_choice, self.model_variant = TrainTask._choose_model()

        mmseg_config_wrapper = self.model_choice(  # pytype: disable=not-instantiable
            self.model_variant,
            train_dataset_task.tile_width,
            train_dataset_task.tile_height,
            train_dataset_task.num_classes, self)
        joblib_dump_atomic(mmseg_config_wrapper,
                           self.mmseg_config_wrapper_file)

    def _get_datasets(
            self,
            eval_done,
            train_done
    ) -> tuple[DatasetTask | None, DatasetTask | None, DatasetTask | None]:
        """
        Gets the dataset(s) for training.

        :param eval_done: Whether evaluation is completed. If separate
            datasets are being used, the test set can be removed after
            evaluation, without affecting task completion
            status.
        :param train_done: Whether training is completed. If separate
            datasets are being used, the train set can be removed after
            training, without affecting task completion
        :return: Tuple of train, val, and test datasets.
        """
        if not train_done or not self.train_dataset_task_dir_ext:
            train_dataset_task = \
                self._get_dataset_task(
                    'Path to dataset:',
                    'train_dataset_task_dir_ext')
            self.train_dataset_task_dir_ext = train_dataset_task.root_dir
        else:
            train_dataset_task = DatasetTask.load(
                self.train_dataset_task_dir_ext, require_done=True)
        if (self.separate_datasets is None and
                (not train_done or not eval_done)):
            self.separate_datasets = \
                prompt_yes_no(
                    "Would you like to provide separate datasets "
                    "for validation and testing?", default=False)
        val_dataset_task, test_dataset_task = None, None
        if self.separate_datasets:
            if not train_done:
                val_dataset_task = self._get_separate_dataset_task(
                    train_dataset_task,
                    'Path to validation dataset:',
                    'val_dataset_task_dir_ext')

            if not eval_done:
                test_dataset_task = self._get_separate_dataset_task(
                    train_dataset_task,
                    'Path to test dataset:',
                    'test_dataset_task_dir_ext')

        return train_dataset_task, val_dataset_task, test_dataset_task  # pytype: disable=bad-return-type

    def _get_num_epochs(self) -> int:
        """
        Gets the number of epochs to train for.

        :return: Number of epochs.
        """
        if not self.num_epochs:
            return prompt_integer(
                "Number of epochs (training iterations "
                "over entire train set):",
                1, None, self.default_num_epochs)

        return self.num_epochs

    def _get_learning_rate(self) -> float:
        """
        Gets the learning rate to use for training.

        :return: Learning rate.
        """
        if not self.learning_rate:
            return prompt_numeric_value(
                "Learning rate:", 0, None, self.default_learning_rate)
        return self.learning_rate

    def _get_device_batch_size(self) -> int:
        """
        Gets the *per-device* batch size to use for training.

        :return: Per-device batch size.
        """
        if not self.device_batch_size:
            return prompt_integer("Batch size:",
                                        1, None,
                                        self.default_device_batch_size)
        return self.device_batch_size

    def _get_num_devices(self) -> int:
        """
        Gets the number of devices to use for training.

        :return: Desired device count.
        """
        if self.num_devices is None:
            dc = torch.cuda.device_count()
            if dc > 0:
                return prompt_integer("Number of GPUs:",
                                            1, dc, self.default_num_devices)
            else:
                warn("No GPUs available, using CPU")
                return self.default_num_devices
        return self.num_devices

    def _get_separate_dataset_task(
            self, train_dataset_task: DatasetTask, prompt: str, field: str
    ) -> DatasetTask:
        """
        Prompts the user for a separate validation/test ``DatasetTask``, and
        checks that it is compatible with the train ``DatasetTask``.

        :param train_dataset_task: ``DatasetTask`` for training.
        :param prompt: Prompt asking for validation or test ``DatasetTask``.
        :param field: Field of ``self`` to write the path of the given
            ``DatasetTask`` to.
        :return: The given ``DatasetTask``.
        """
        while True:
            dataset_task = self._get_dataset_task(prompt, field)
            if not UnlabeledDatasetTask.compare_tile_size(
                    train_dataset_task, dataset_task):
                continue
            if ((dataset_task.num_classes !=
                 train_dataset_task.num_classes) or (
                    dataset_task.label_legend !=
                    train_dataset_task.label_legend)):
                warn("Given dataset is incompatible: set of classes differs "
                     "from train dataset")
                continue
            break
        self.__setattr__(field, dataset_task.root_dir)
        return dataset_task

    def _get_dataset_task(self, prompt: str,
                          field: str) -> DatasetTask:
        """
        Prompts the user for a ``DatasetTask``.

        :param prompt: Prompt for the user.
        :param field: Field of ``self`` to write the path of the given
             ``DatasetTask`` to.
        :return: The given ``DatasetTask``.
        """
        getattr(self, field)

        if not self.__dict__[field]:
            dataset_task_file, dataset_task = \
                prompt_with_predicates(
                    prompt,
                    [(lambda x: DatasetTask.try_load(x), "")])
            dataset_task = dataset_task[0]
        else:
            dataset_task = DatasetTask.load(self.__dict__[field])

        return dataset_task

    def _make_train_val_test_split(self,
                                   train_dataset_task: DatasetTask,
                                   val_dataset_task: DatasetTask | None,
                                   test_dataset_task: DatasetTask | None,
                                   split: tuple[float, float],
                                   separate_datasets: bool) -> None:
        """
        Calculates the train-val-test split.

        :param train_dataset_task: ``DatasetTask`` to train on.
        :param val_dataset_task: ``DatasetTask`` to validate on.
        :param test_dataset_task: ``DatasetTask`` to test on.
        :param split: Tuple containing the absolute proportion of the train
            set, and the proportion of the train set to use for validation.
        :param separate_datasets: Whether separate datasets are used for
            validation and testing.

        Examples
        --------
        With ``split=(0.8, 0.2)``::

            test_fraction = 1 - 0.8 = 0.2
            val_fraction = 0.8 * 0.2 = 0.16.
            train_fraction = 0.8 - 0.16 = 0.64.
        """
        assert 0 < split[0] < 1 and 0 < split[1] < 1, \
            "Train and validation set fractions must be in the range (0, 1)"

        # The split depends on whether 1 or 3 datasets were given.
        # In the separate case, shuffle the train set and use all images
        # of each set.
        if separate_datasets:
            train_names = get_stems(
                train_dataset_task.image_tiles_dir)

            sample_weights_journal = Journal(
                train_dataset_task.sample_weights_journal_file)
            train_sample_weights = list(sample_weights_journal.values())

            train_names = self._shuffle_train_and_sample_weights(
                train_names, train_sample_weights)

            val_names = get_stems(
                val_dataset_task.image_tiles_dir)  # pytype: disable=attribute-error

            test_names = get_stems(
                test_dataset_task.image_tiles_dir)  # pytype: disable=attribute-error

        # If one dataset is given, split the dataset along the horizontal line
        # and shuffle only the train portion after splitting.
        else:
            filename_stubs = get_stems(
                train_dataset_task.image_tiles_dir)

            train_length = int(len(filename_stubs) * split[0])
            val_length = int(train_length * split[1])

            train_names = filename_stubs[:train_length - val_length]

            sample_weights_journal = Journal(
                train_dataset_task.sample_weights_journal_file)

            # Adjust the sample_weights from the dataset to only contain
            # the weights for train samples
            train_sample_weights = list(sample_weights_journal.values()
                                        )[:train_length - val_length]

            train_names = self._shuffle_train_and_sample_weights(
                train_names, train_sample_weights)

            val_names = filename_stubs[train_length - val_length:train_length]

            test_names = filename_stubs[train_length:]

        self._write_splits(train_names, val_names, test_names)

    def _shuffle_train_and_sample_weights(self, train_names: list[str],
                                          train_sample_weights: list[float]):
        """
        Shuffles the samples in the training set together with their associated
        weights, to ensure that after shuffling the sample weights are still
        correctly aligned with the training samples.

        :param train_names: Filenames in the training set.
        :param train_sample_weights: Sample weights for training set.
        :return: Shuffled training names. The sample weights are written to
            disk in compressed joblib format.
        """
        assert len(train_sample_weights) == len(train_names)

        names_weights = list(zip(train_names, train_sample_weights))
        shuffle(names_weights)
        train_names, train_sample_weights = map(list, zip(*names_weights))

        sample_weights_file = os.path.join(
            self.root_dir, "sample_weights.joblib")
        joblib_dump_atomic(train_sample_weights, sample_weights_file)
        self.sample_weights_file = sample_weights_file

        return train_names

    def _write_splits(self, train_names, val_names, test_names) -> None:
        """
        Writes the given train, validation, and test splits to disk.

        :param train_names: Files in training set.
        :param val_names: Files in validation set.
        :param test_names: Files in test set.
        """
        list_to_file(
            train_names,
            self.train_split_file)

        list_to_file(
            val_names,
            self.val_split_file)

        list_to_file(
            test_names,
            self.test_split_file)

    @staticmethod
    def _choose_model() -> tuple[type[MMSegConfigWrapper], str]:
        """
        Gets the class of the model to train with, based on the user's choice.

        :return: Tuple with class inheriting from ``MMSegConfigWrapper``, and
            a path to the config for the chosen model variant.
        """
        print("Please choose a model for training:")

        adaptors = MMSegConfigWrapper.__subclasses__()

        for i, model in enumerate(adaptors):
            print(f"{i + 1}. {model.get_model_name()}")

        choice = prompt_integer("Choice:", 1, len(adaptors))
        model_wrapper = adaptors[choice - 1]
        variants = model_wrapper.get_model_variants()

        choice = TrainTask._choose_variant(variants)
        return model_wrapper, variants[choice]["Name"]

    @staticmethod
    def _choose_variant(variants) -> int:
        """
        Prompts the user to choose a variant from a list of variants.

        :param variants: A list of dictionaries, each containing the 'Name'
            and 'Metadata' of a variant.
        :return: The chosen variant, or None if the user chose to exit.
        """

        print("\nPlease choose a variant for this model:")

        num_variants = len(variants)
        variants_per_page = 3
        pages = num_variants // variants_per_page + 1

        for page in range(pages):
            start_index = page * variants_per_page
            end_index = start_index + variants_per_page
            for i, variant in enumerate(variants[start_index:end_index],
                                        start=start_index + 1):
                print(f'{i}. {variant["Name"]}')
                TrainTask._print_variant_metadata(variant)
            if page < pages - 1:
                choice = prompt_integer(
                    "Press ENTER to see more options, or type your choice:",
                    low=1, high=num_variants, default=-1, hide_default=True)
                if choice == -1:
                    # Move cursor up one line and to the start, so that
                    # the next print batch erases the input prompt
                    sys.stdout.write('\x1b[A\x1b[K')
                    continue
                return choice
            else:
                return prompt_integer(
                    "Choice:", low=1, high=num_variants
                )
        raise RuntimeError("No model choices available")

    @staticmethod
    def _print_variant_metadata(variant: dict) -> None:
        """
        Prints metadata for a given model variant.

        :param variant: A model variant.
        """
        m = variant["Metadata"]
        if 'Training Data' in m:
            print(f'\tTraining Data:\t{m["Training Data"]}')
        if 'Architecture' in m:
            print(f'\tArchitecture:\t{m["Architecture"]}')
        if 'Memory (GB)' in m:
            print(f'\tMemory (GB):\t{m["Memory (GB)"]}')
        print()

    def is_training_done(self) -> bool:
        """
        Checks whether the model is done training.

        :return: ``True`` if the required training files exist and the latest
            model checkpoint has an iteration number equal to or greater than
            the configured maximum training iterations.
        """
        if not os.path.exists(self.mmseg_config_wrapper_file):
            return False

        mmseg_config_wrapper = joblib_load_typed(
            self.mmseg_config_wrapper_file,
            MMSegConfigWrapper)

        latest_checkpoint_file = \
            mmseg_config_wrapper.try_get_latest_checkpoint(
                self.model_working_dir)

        if latest_checkpoint_file is None:
            return False

        # E.g., iter_503
        latest_checkpoint = Path(latest_checkpoint_file).stem
        if 'iter' not in latest_checkpoint:
            return False

        latest_checkpoint = int(latest_checkpoint.split('_')[1])

        if (latest_checkpoint >=
                mmseg_config_wrapper.mmseg_config.train_cfg.max_iters):
            return True

        return False

    def is_evaluation_done(self) -> bool:
        """
        Checks whether the model has been evaluated.

        :return: ``True`` if ``self.eval`` is ``True`` and the "test"
            directory exists in ``self.model_working_dir``.
            ``False`` otherwise.
        """
        if self.eval_done and not os.path.exists(os.path.join(
                self.model_working_dir, 'test')):
            self.eval_done = False
        return self.eval_done

    def is_done(self) -> bool:
        return self.is_training_done() and self.is_evaluation_done()

    def _run(self):
        if not self.is_training_done():
            print(flush=True)
            print("Launching model training ...")
            sleep(1)
            train(self)
        if not self.is_evaluation_done():
            print(flush=True)
            print("Launching model evaluation ...")
            sleep(1)
            evaluate(self)

    def train_test_split_done(self):
        """
        Checks whether the train-val-test split has been computed.

        :return: ``True`` if the three text files containing the filenames for
            each split and the sample weights exist on disk, ``False``
            otherwise.
        """
        return (os.path.exists(self.split_lists_dir)
                and os.path.exists(self.train_split_file)
                and os.path.exists(self.val_split_file)
                and os.path.exists(self.test_split_file)
                and os.path.exists(self.sample_weights_file))

    def get_model_data(self) -> tuple[str, MMSegConfigWrapper, DatasetTask]:
        """
        Loads principal model data.

        :return: Tuple of MMSegmentation config filepath,
            ``MMSegConfigWrapper``, and ``DatasetTask`` for training.
        """
        train_dataset_task = DatasetTask.load(
            self.train_dataset_task_dir_ext)
        if train_dataset_task is None:
            raise RuntimeError(
                "Failed to load dataset from "
                f"'{self.train_dataset_task_dir_ext}'"
            )
        mmseg_config_wrapper = joblib_load_typed(
            self.mmseg_config_wrapper_file,
            MMSegConfigWrapper)
        mmseg_config_file = mmseg_config_wrapper.try_get_mmseg_config_file(
            self.model_working_dir)

        if not mmseg_config_file:
            raise RuntimeError("No MMSegmentation config found for training, "
                               "although it should have been downloaded")

        return mmseg_config_file, mmseg_config_wrapper, train_dataset_task


def train(train_task: TrainTask) -> None:
    """
    Entry point for model training.

    :param train_task: Train task.
    """
    mmseg_config_file, mmseg_config_wrapper, train_dataset_task = \
        train_task.get_model_data()

    mmseg_config_wrapper.mmseg_config.launcher = \
        "pytorch" if is_running_with_multiple_gpus() else "none"

    latest_checkpoint_file = mmseg_config_wrapper.try_get_latest_checkpoint(
        train_task.model_working_dir)
    if not latest_checkpoint_file:
        warn("No checkpoint was loaded, model will not be pretrained!")
        sleep(1)
    else:
        if 'iter' in Path(latest_checkpoint_file).stem:
            print(f"Resuming training from latest checkpoint: "
                  f"{Path(latest_checkpoint_file).stem}")
            mmseg_config_wrapper.mmseg_config.resume = True
        else:
            mmseg_config_wrapper.mmseg_config.resume = False

    # Update the mmseg config before launching the training process
    mmseg_config_wrapper.dump_mmseg_config(mmseg_config_file)

    if is_running_with_multiple_gpus():
        here = os.path.dirname(os.path.abspath(__file__))
        env = os.environ.copy()
        port = 29500 if "PORT" not in env else env["PORT"]

        subprocess.run(
            ['python', '-m',
             'torch.distributed.launch',
             f'--nproc_per_node={train_task.num_devices}',
             f'--master_port={port}',
             os.path.join(here, '../launchers/train_launcher.py'),
             train_task.root_dir,
             train_dataset_task.root_dir],
            check=True)
    else:
        register_train_components(train_task, train_dataset_task)
        _train(train_task, mmseg_config_wrapper.mmseg_config)


def evaluate(train_task: TrainTask) -> None:
    """
    Entry point for model evaluation.

    :param train_task: Train task.
    """
    mmseg_config_file, mmseg_config_wrapper, train_dataset_task = \
        train_task.get_model_data()

    mmseg_config_wrapper.mmseg_config.launcher = \
        "pytorch" if is_running_with_multiple_gpus() else "none"
    mmseg_config_wrapper.mmseg_config.resume = False
    mmseg_config_wrapper.dump_mmseg_config(mmseg_config_file)

    if is_running_with_multiple_gpus():
        here = os.path.dirname(os.path.abspath(__file__))
        env = os.environ.copy()
        subprocess.run(
            ['python', '-m',
             'torch.distributed.launch',
             f'--nproc_per_node={train_task.num_devices}',
             f'--master_port={29500 if "PORT" not in env else env["PORT"]}',
             os.path.join(here, '../launchers/eval_launcher.py'),
             train_task.root_dir,
             train_dataset_task.root_dir],
            check=True)
    else:
        register_eval_components(train_task, train_dataset_task)
        _evaluate(train_task, mmseg_config_file)
        train_task.eval_done = True


def _train(train_task: TrainTask, mmseg_config: Config):
    """
    Threadsafe portion of training logic that can run on multiple devices.
    This method is accessed by ``TrainTask.train`` if 1 device is used,
    or by each device via ``ecomapper.launchers.train_launcher`` otherwise.

    :param train_task: Train task.
    :param mmseg_config: MMSegmentation model configuration.
    """
    runner = Runner.from_cfg(mmseg_config)
    assert runner.seed == train_task.rng_seed
    runner.train()
    deregister_all()


def _evaluate(train_task: TrainTask, mmseg_config_file: str):
    """
    Threadsafe portion of evaluation logic that can run on multiple devices.
    This method is accessed by ``TrainTask.evaluate`` if 1 device is used,
    or by each device via ``ecomapper.launchers.eval_launcher`` otherwise.

    :param train_task: Train task.
    :param mmseg_config_file: Filepath to MMSegmentation model configuration.
    """
    best_checkpoint_file = get_unique_file_in_directory(
        train_task.model_working_dir,
        'best_*.pth')

    if not best_checkpoint_file:
        best_checkpoint_file = MMSegConfigWrapper.try_get_latest_checkpoint(
            train_task.model_working_dir)
        if not best_checkpoint_file:
            raise RuntimeError("Cannot evaluate model: "
                               "no checkpoint available to load from")

    old_argv = sys.argv
    sys.argv = [os.path.basename(__file__),
                mmseg_config_file,
                best_checkpoint_file,
                "--work-dir",
                os.path.join(
                    train_task.model_working_dir,
                    'test')]
    sys.argv += ["--launcher", "pytorch"] \
        if is_running_with_multiple_gpus() \
        else []
    mmseg_test()
    sys.argv = old_argv
    deregister_all()
