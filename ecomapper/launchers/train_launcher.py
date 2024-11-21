"""
Distributed training launcher.
"""

import sys

from mmengine import Config

from ecomapper.mmseg_interface.mmseg_config_wrapper import MMSegConfigWrapper
from ecomapper.mmseg_interface.registry import register_train_components
from ecomapper.task.dataset_task import DatasetTask
from ecomapper.task.train_task import TrainTask
from ecomapper.task.train_task import _train
from ecomapper.utils.joblib_util import joblib_load_typed


def run_train_from_args() -> None:
    """
    Executes the ``_train`` function of ``TrainTask`` when running
    in distributed mode with two or more GPUs.
    """
    task = TrainTask.load(sys.argv[2], require_done=False, read_only=True)
    train_dataset_task = DatasetTask.load(sys.argv[3], read_only=True)

    register_train_components(task, train_dataset_task)

    mmseg_config_wrapper = joblib_load_typed(task.mmseg_config_wrapper_file,
                                             MMSegConfigWrapper)
    mmseg_config_file = mmseg_config_wrapper.try_get_mmseg_config_file(
        task.model_working_dir)
    if not mmseg_config_file:
        raise RuntimeError("No MMSegmentation config found for training, "
                           "although it should have been downloaded")
    mmseg_config = Config.fromfile(mmseg_config_file)

    _train(task, mmseg_config)


if __name__ == "__main__":
    run_train_from_args()
