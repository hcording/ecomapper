"""
Distributed inference launcher.
"""

import sys

from ecomapper.core.journal import Journal
from ecomapper.mmseg_interface.registry import deregister_all
from ecomapper.mmseg_interface.registry import register_predict_components
from ecomapper.task.dataset_task import DatasetTask
from ecomapper.task.predict_task import PredictTask
from ecomapper.task.predict_task import _predict
from ecomapper.task.train_task import TrainTask


def run_predict_from_args() -> None:
    """
    Executes the ``_predict`` function of ``PredictTask`` when running
    in distributed mode with two or more GPUs.
    """
    predict_task = PredictTask.load(sys.argv[2],
                                        require_done=False, read_only=True)
    train_task = TrainTask.load(sys.argv[3],
                                    require_done=False, read_only=True)
    train_dataset_task = DatasetTask.load(
        train_task.train_dataset_task_dir_ext, read_only=True)

    predict_journal = Journal(predict_task.predict_journal_file)

    register_predict_components(train_task,
                                train_dataset_task, predict_journal)

    _predict(predict_task, train_task)

    deregister_all()


if __name__ == "__main__":
    run_predict_from_args()
