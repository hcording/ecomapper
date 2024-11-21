"""
Execution of all Tasks in sequence to produce a trained model which can
repeatedly produce predictions from new inputs.
"""

import os
from typing import Type

from ecomapper.task.dataset_task import DatasetTask
from ecomapper.task.predict_task import PredictTask
from ecomapper.task.task import Task
from ecomapper.task.train_task import TrainTask
from ecomapper.utils.date_util import convert_to_humanly_readable_date
from ecomapper.utils.date_util import filename_friendly_date
from ecomapper.utils.path_util import make_dir
from ecomapper.utils.prompt_util import prompt_yes_no


class PipelineTask(Task):
    def __init__(self,
                 separate_datasets: bool | None = None,
                 predict_task_dir: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.separate_datasets = separate_datasets
        """
        Whether to use separate datasets for training, validation, and testing.
        """

        self.train_dataset_task_dir = os.path.join(self.root_dir,
                                                   'train_dataset')
        """
        Directory to store the dataset for training in.
        """

        self.val_dataset_task_dir = os.path.join(self.root_dir,
                                                 'val_dataset')
        """
        Directory to store the dataset for validation in.
        Only used if ``self.separate_datasets`` is ``True``.
        """

        self.test_dataset_task_dir = os.path.join(self.root_dir,
                                                  'test_dataset')
        """
        Directory to store the dataset for testing in.
        Only used if ``self.separate_datasets`` is ``True``.
        """

        self.train_task_dir = os.path.join(self.root_dir, 'model')
        """
        Directory to store the model in.
        """

        self.predict_task_dir = predict_task_dir
        """
        Directory to store predictions in.
        When a prediction is completed, this is set to ``None``, and next
        time the pipeline is started, a new ``PredictTask`` will be
        created to allow reapplication of the trained model.
        """

    def _get_input(self) -> None:
        if self.separate_datasets is None:
            self.separate_datasets = prompt_yes_no(
                "Would you like to provide separate datasets "
                "for validation and testing?", default=False)

    def make_task(self, field: str, task_type: Type[Task.T],
                  run: bool = True) -> Task.T:
        """
        Creates a Task for use in the pipeline.

        :param field: Field of ``self`` to store the directory of the created
            Task in.
        :param task_type: Type of the Task being created.
        :param run: Whether to execute the Task after creation.
        :return: Instantiated Task.
        """
        # Sanity check that this field exists
        getattr(self, field)

        if not os.path.exists(self.__dict__[field]):
            print(f"Creating new {task_type.__name__} ...\n")
            task_dir = make_dir(os.path.join(
                self.root_dir, self.__dict__[field]))
            task = task_type(
                creation_date=filename_friendly_date(),
                app_name=self.app_name,
                app_version=self.app_version,
                root_dir=task_dir,
                rng_seed=self.rng_seed,
                verbose=self.verbose)
        else:
            task_dir = self.__dict__[field]
            task = task_type.load(self.__dict__[field],
                                      require_done=False)
            date = convert_to_humanly_readable_date(task.creation_date)
            print(f"Loading existing \"{task.__class__.__name__}\", "
                  f"created on {date} ...\n")
        if task_type != PredictTask:
            self.__setattr__(field, task_dir)

        if run:
            task.run()

        return task

    def is_done(self) -> bool:
        """
        The ``PipelineTask`` is never done.
        It executes all other tasks in sequence, which will terminate early
        if they are done.
        Lastly, the trained model can be used repeatedly to make predictions.

        :return: ``False``.
        """
        return False

    def _run(self) -> None:
        # Get the datasets for training
        train_dataset_task = \
            self.make_task('train_dataset_task_dir',
                           DatasetTask)
        if self.separate_datasets:
            val_dataset_task = \
                self.make_task('val_dataset_task_dir',
                               DatasetTask)
            test_dataset_task = \
                self.make_task('test_dataset_task_dir',
                               DatasetTask)

        # Create and configure the training Task
        train_task = self.make_task('train_task_dir', TrainTask, run=False)
        train_task.separate_datasets = self.separate_datasets
        train_task.train_dataset_task_dir_ext = train_dataset_task.root_dir
        if self.separate_datasets:
            train_task.val_dataset_task_dir_ext = val_dataset_task.root_dir
            train_task.test_dataset_task_dir_ext = test_dataset_task.root_dir

        # Train a model
        train_task.run()

        # Create a new directory for predictions, if desired
        if (self.predict_task_dir is None
                and prompt_yes_no(
                    "Would you like to use the trained model "
                    "to make predictions on a dataset?", False)):
            self.predict_task_dir = (
                    'predictions' + f'_{filename_friendly_date()}')

        # Make predictions on a given dataset, if desired
        if self.predict_task_dir is not None:
            predict_task = self.make_task('predict_task_dir',
                                          PredictTask,
                                          run=False)
            predict_task.train_task_dir_ext = train_task.root_dir
            predict_task.run()

            # Remove association so a new ``PredictTask`` can be run next time
            self.predict_task_dir = None
