"""
Predictions with an MMSegmentation model.
"""

import os
import subprocess
import sys
from glob import glob
from typing import Any

from ecomapper.core.journal import Journal
from ecomapper.core.merge import merge_predictions
from ecomapper.core.merge import write_segmentation_map
from ecomapper.launchers.mmseg_test import main as mmseg_test
from ecomapper.mmseg_interface.mmseg_config_wrapper import MMSegConfigWrapper
from ecomapper.mmseg_interface.registry import deregister_all
from ecomapper.mmseg_interface.registry import register_predict_components
from ecomapper.task.dataset_task import DatasetTask
from ecomapper.task.task import Task
from ecomapper.task.train_task import TrainTask
from ecomapper.task.unlabeled_dataset_task import UnlabeledDatasetTask
from ecomapper.utils.date_util import filename_friendly_date
from ecomapper.utils.joblib_util import joblib_load_typed
from ecomapper.utils.path_util import make_dir
from ecomapper.utils.print_util import warn
from ecomapper.utils.prompt_util import prompt_with_predicates
from ecomapper.utils.torch_util import is_running_with_multiple_gpus


class PredictTask(Task):

    def __init__(self,
                 train_task_dir_ext: str | None = None,
                 dataset_task_dir_ext: str | None = None,
                 segmentation_map_file: str | None = None,
                 **kwargs: dict[str, Any]):
        super().__init__(**kwargs)

        self.train_task_dir_ext = train_task_dir_ext
        """
        Path to a directory containing a ``TrainTask``.
        The trained model contained in this task will be used to make
        predictions.
        """

        self.dataset_task_dir_ext = dataset_task_dir_ext
        """
        Path to a directory containing a ``DatasetTask`` or
        ``UnlabeledDatasetTask``. The image tiles in this dataset will be fed
        to the model for inference.
        """

        self.model_working_dir = make_dir(os.path.join(
            self.root_dir, 'model_working_dir'))
        """
        Working directory for the MMSegmentation model during inference.
        """

        self.predict_journal_file = self.register_journal('predict_journal')
        """
        Tracks for which image tiles from the input dataset the model has
        already made predictions.
        """

        self.prediction_tiles_dir = make_dir(os.path.join(
            self.root_dir, 'prediction_tiles'))
        """
        Directory in which to store prediction tiles.
        """

        self.merge_journal_file = self.register_journal('merge_journal')
        """
        Tracks which predicted tiles have already been merged into
        ``self.merged_predictions_file``.
        """

        self.merged_predictions_file = os.path.join(
            self.root_dir, 'merged_predictions.dat')
        """
        Numpy memory mapping with dimensions as given by the original image
        metadata of the input dataset. All prediction tiles are merged into
        this array.
        """

        self.segmentation_map_file = segmentation_map_file
        """
        A .tif image with dimensions as given by the original image
        metadata of the input dataset. This image is the final segmentation
        map that can be overlaid onto the original image to visualize the model
        predictions.
        """

    def _get_input(self):
        # First obtain the model for predicting
        if not self.train_task_dir_ext:
            while True:
                train_task_dir_ext, train_task = prompt_with_predicates(
                    "Path to trained model:",
                    [(lambda x: TrainTask.try_load(
                        x, require_done=False), "")])
                train_task = train_task[0]
                if not MMSegConfigWrapper.try_get_latest_checkpoint(
                        train_task.model_working_dir):
                    warn("No weight file was found for this model "
                         "(is it trained yet?)")
                else:
                    break
            self.train_task_dir_ext = train_task_dir_ext
        else:
            train_task = TrainTask.load(
                self.train_task_dir_ext, require_done=False)

        if not train_task.is_training_done():
            warn("The selected model is not fully trained -- "
                 "prediction quality might be low")

        # Now get a dataset to make predictions on
        if not self.dataset_task_dir_ext:
            train_dataset_task = DatasetTask.load(
                train_task.train_dataset_task_dir_ext)
            while True:
                dataset_task_dir_ext, dataset_task = prompt_with_predicates(
                    "Path to dataset:",
                    [(lambda x: UnlabeledDatasetTask.try_load(x), "")])
                dataset_task = dataset_task[0]

                # Input size of the given dataset must be the same as
                # in the model training dataset
                if not UnlabeledDatasetTask.compare_tile_size(
                        train_dataset_task, dataset_task):
                    continue
                break

            self.dataset_task_dir_ext = dataset_task_dir_ext
        else:
            UnlabeledDatasetTask.load(self.dataset_task_dir_ext)

    def is_done_predict(self) -> bool:
        """
        Checks whether the model inference is completed.

        :return: ``True`` if all predictions are done, ``False`` otherwise.
        """
        # Predictions require a dataset
        if self.dataset_task_dir_ext is None:
            return False

        # Get the Journal and see if it's complete
        predict_journal = Journal(self.predict_journal_file)
        dataset_task = UnlabeledDatasetTask.load(self.dataset_task_dir_ext)
        return predict_journal.is_done(dataset_task.image_tiles_dir)

    def is_done_merge(self) -> bool:
        """
        Checks whether the model predictions have been merged into a
        segmentation map.

        :return: :code:`True` if the segmentation map has been created and the
            Journal is complete.
        """
        # Cannot be done if predictions are not finished
        if not self.is_done_predict():
            return False

        # Cannot be done if the segmentation map or raw data of merged
        # predictions do not exist
        if (self.segmentation_map_file is None or not os.path.exists(
                self.segmentation_map_file)) or (
                not os.path.exists(self.merged_predictions_file)):
            return False

        dataset_task = DatasetTask.load(self.dataset_task_dir_ext)

        # Get the Journal and see if it's complete
        merge_journal = Journal(self.merge_journal_file)
        return ('iter' in merge_journal
                and merge_journal['iter'][2] == dataset_task.total_num_tiles
                and self.segmentation_map_file is not None
                and os.path.exists(self.segmentation_map_file))

    def is_done(self) -> bool:
        return self.is_done_predict() and self.is_done_merge()

    def _run(self):
        predict(self)


def predict(predict_task: PredictTask) -> None:
    """
    Entry point for model inference.

    :param predict_task: Predict task.
    """
    predict_journal = Journal(predict_task.predict_journal_file)
    dataset_task = UnlabeledDatasetTask.load(
        predict_task.dataset_task_dir_ext)

    remaining_tiles = predict_journal.get_remaining(
        dataset_task.image_tiles_dir)

    # Make a file containing the names of remaining files.
    # MMSegmentation will use this file to know which tiles to run the model
    # on.
    ann_file = os.path.join(predict_task.root_dir, 'tiles.txt')
    with open(ann_file, 'w+') as f:
        for r in remaining_tiles:
            f.write(os.path.splitext(r)[0] + '\n')

    train_task = TrainTask.load(predict_task.train_task_dir_ext,
                                    require_done=False)

    # Borrow the model config from the train Task, adjust it, and save it
    # the working directory of the prediction Task
    if not MMSegConfigWrapper.try_get_mmseg_config_file(
            predict_task.model_working_dir):
        setup_mmseg_config(ann_file, predict_task, train_task,
                           dataset_task.image_tiles_dir)

    # Make remaining predictions
    if not predict_task.is_done_predict():
        if len(predict_journal) > 0:
            print(f"Continuing predictions "
                  f"from tile number {len(predict_journal)}")

        # Either launch training normally if 1 device is available,
        # or use the PyTorch distributed launcher
        if is_running_with_multiple_gpus():
            here = os.path.dirname(os.path.abspath(__file__))
            env = os.environ.copy()
            port = 29500 if "PORT" not in env else env["PORT"]
            subprocess.run(
                ['python', '-m',
                 'torch.distributed.launch',
                 f'--nproc_per_node={train_task.num_devices}',
                 f'--master_port={port}',
                 os.path.join(here, '../launchers/predict_launcher.py'),
                 predict_task.root_dir,
                 train_task.root_dir],
                check=True)
        else:
            train_dataset_task = DatasetTask.load(
                train_task.train_dataset_task_dir_ext)

            register_predict_components(train_task,
                                        train_dataset_task, predict_journal)
            _predict(predict_task, train_task)

    # Merge model predictions
    if not predict_task.is_done_merge():
        merge_predictions(
            predict_task.merged_predictions_file,
            predict_task.prediction_tiles_dir,
            train_task.SCALE is not None,
            Journal(predict_task.merge_journal_file),
            list(Journal(dataset_task.fragment_journal_file).values()),
            dataset_task.image_metadata,
            dataset_task.tile_width,
            dataset_task.tile_height)

    # Create a segmentation map from model predictions
    if predict_task.segmentation_map_file is None or not os.path.exists(
            predict_task.segmentation_map_file):
        now = filename_friendly_date()
        segmentation_map_file = os.path.join(predict_task.root_dir,
                                             f"segmentation_map_{now}.tif")
        assert not os.path.exists(segmentation_map_file), \
            f"Cannot create segmentation map, " \
            f"file exists: {segmentation_map_file}"

        write_segmentation_map(predict_task.merged_predictions_file,
                               segmentation_map_file,
                               dataset_task.image_metadata)

        predict_task.segmentation_map_file = segmentation_map_file

        print(f"The final segmentation map has "
              f"been written to: {segmentation_map_file}")


def _predict(predict_task: PredictTask, train_task: TrainTask) -> None:
    """
    Threadsafe portion of prediction logic that can run on multiple devices.
    This method is accessed by ``PredictTask.predict`` if 1 device is used,
    or by each device via ``ecomapper.launchers.predict_launcher`` otherwise.

    :param predict_task: Predict task.
    :param train_task: Train task providing model for inference.
    """
    # Identify the best checkpoint to predict with
    best_checkpoint_files = glob(os.path.join(
        train_task.model_working_dir, "best*.pth"))
    assert len(best_checkpoint_files) == 1, \
        f"More than 1 best checkpoint: {best_checkpoint_files}"
    best_checkpoint_file = best_checkpoint_files[0]

    # Ensure config availability
    mmseg_config_file = MMSegConfigWrapper.try_get_mmseg_config_file(
        predict_task.model_working_dir)
    assert mmseg_config_file is not None, \
        "No model config file found, although it should have been created"

    # Launch MMSegmentation inference
    old_argv = sys.argv
    sys.argv = [os.path.basename(__file__),
                mmseg_config_file,
                best_checkpoint_file,
                "--work-dir", predict_task.model_working_dir,
                "--out", predict_task.prediction_tiles_dir]
    sys.argv += ["--launcher", "pytorch"] \
        if is_running_with_multiple_gpus() \
        else []
    mmseg_test()
    sys.argv = old_argv
    deregister_all()


def setup_mmseg_config(ann_file: str, predict_task: PredictTask,
                       train_task: TrainTask,
                       image_tiles_dir_ext: str) -> None:
    """
    Opens the MMSegmentation configuration of ``train_task``, modifies it
    for inference, and saves it to ``predict_task.model_working_dir``.

    :param ann_file: Text file containing filenames of image tiles to run
        prediction on.
    :param predict_task: Predict task.
    :param train_task: Train task providing model for inference.
    :param image_tiles_dir_ext: Directory containing image tiles.
    """
    mmseg_config_wrapper = \
        joblib_load_typed(train_task.mmseg_config_wrapper_file,
                          MMSegConfigWrapper)
    mmseg_config = mmseg_config_wrapper.mmseg_config

    mmseg_config.work_dir = predict_task.model_working_dir
    mmseg_config.train_pipeline = None
    mmseg_config.train_dataloader.dataset.pipeline = None
    mmseg_config.test_evaluator.format_only = True
    mmseg_config.test_dataloader.dataset.data_prefix = \
        dict(img_path=image_tiles_dir_ext)
    mmseg_config.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='EvalScaleAugmentation'),
        dict(type='PackSegInputs')]
    mmseg_config.test_dataloader.dataset.pipeline = mmseg_config.test_pipeline
    mmseg_config.test_dataloader.dataset.ann_file = ann_file
    mmseg_config.resume = False
    mmseg_config.custom_hooks = [
        dict(type='InferenceHook',
             priority='ABOVE_NORMAL')]

    mmseg_config.launcher = "pytorch" \
        if is_running_with_multiple_gpus() \
        else "none"

    mmseg_config_file = os.path.join(
        predict_task.model_working_dir,
        os.path.basename(mmseg_config.filename))

    mmseg_config_wrapper.dump_mmseg_config(mmseg_config_file)
