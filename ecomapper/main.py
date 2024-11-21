"""
EcoMapper entry point.
"""

import argparse
import importlib
import os
import sys
from typing import Type

from osgeo import gdal

from ecomapper.task.task import Task
from ecomapper.utils import prompt_util
from ecomapper.utils.date_util import convert_to_humanly_readable_date
from ecomapper.utils.date_util import filename_friendly_date
from ecomapper.utils.path_util import expand_path
from ecomapper.utils.print_util import success
from ecomapper.utils.random_util import set_seed

gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()


def get_app_name_and_version() -> tuple[str, str]:
    """
    Returns the app name and version.
    :return: Tuple of app name, version.
    """
    return "EcoMapper", "1.0.0"


def get_task(choice: str) -> tuple[str, str]:
    """
    Translates the application mode chosen by the user to a Task name.

    :param choice: User choice.
    :return: Tuple of Task module name, Task class name.
    """
    choice_to_task = {
        'labeled-dataset': ('dataset_task', 'DatasetTask'),
        'unlabeled-dataset': ('dataset_task', 'UnlabeledDatasetTask'),
        'train': ('train_task', 'TrainTask'),
        'predict': ('predict_task', 'PredictTask'),
        'pipeline': ('pipeline_task', 'PipelineTask')
    }

    return choice_to_task[choice]


def main() -> None:
    """
    EcoMapper application.
    """
    name, version = get_app_name_and_version()
    print(name, version, '\n')

    seed = 999
    """
    The random seed for the application.
    This seed is set for all major
    random libraries, including ``random``, ``np.random``, and ``torch``.
    MMSegmentation uses the seed for deterministic training and evaluation
    of models.
    """

    set_seed(seed)
    verbose = True

    args = parse_args()
    prompt_util.ASSUME_DEFAULT = args.assume_default

    print("Initializing ...")

    # This imports the required task dynamically to reduce loading time
    task_module_name, task_class_name = get_task(args.task)
    package_name = 'ecomapper.task'
    module = importlib.import_module(
        f"{package_name}.{task_module_name}", task_class_name)
    task_class: Type[Task.T] = getattr(module, task_class_name)

    working_dir: str = expand_path(args.directory)
    task_file = os.path.join(working_dir, Task.TASK_FILENAME)

    # CVAT labeling logic
    if args.task == 'labeled-dataset' and args.with_cvat:
        UnlabeledDatasetTask = importlib.import_module(
            'ecomapper.task.unlabeled_dataset_task').UnlabeledDatasetTask

        DatasetTask = importlib.import_module(
            'ecomapper.task.dataset_task').DatasetTask

        # 1. Split the input image using an ``UnlabeledDatasetTask``
        task = run_task(UnlabeledDatasetTask, name, seed, version,
                        task_file, verbose, working_dir)

        if task.__class__ == DatasetTask:
            # We have already created a ``DatasetTask`` with the labels
            # from CVAT and ran it above, so nothing left to do
            return

        print("\nThe image has been split, "
              f"tiles are located in: {task.image_tiles_dir}")
        print("Please now visit https://cvat.ai to "
              "upload the tiles and label them")
        print("N.B.: When downloading the labels, "
              "choose the 'Cityscapes 1.0' format")
        if not args.assume_default:
            input("Press any key to continue: ")

        # 2. Using CVAT labels, mutate the Task into a ``DatasetTask``
        DatasetTask.with_cvat(task)
        return

    run_task(task_class, name, seed,
             version, task_file, verbose, working_dir)


def run_task(task_class: Type[Task.T], name: str, seed: int, version: str,
             task_file: str, verbose: bool, working_dir: str) -> Task.T:
    """
    Creates or loads the task at ``task_file`` with type ``task_class``.

    :param task_class: Type of the Task.
    :param name: App name.
    :param seed: Random seed.
    :param version: App version.
    :param task_file: Filepath to Task.
    :param verbose: Verbosity.
    :param working_dir: Working directory given by the user.
    :return: Finished Task.
    """
    if not os.path.exists(working_dir) \
            or not os.path.isfile(task_file):

        if verbose:
            print(f"Creating new {task_class.__name__} ...\n")

        task = make_basic_task(task_class, name, working_dir,
                               seed, verbose, version)
    else:
        task = task_class.load(working_dir,
                                   require_done=False, verbose=verbose)
        if task is None:
            sys.exit(1)
        if verbose:
            date = convert_to_humanly_readable_date(task.creation_date)
            print(f"Loading existing \"{task.__class__.__name__}\", "
                  f"created on {date} ...\n")

    task.run()
    return task


def make_basic_task(task_class: Type[Task.T], name: str, working_dir: str,
                    seed: int, verbose: bool, version: str) -> Task.T:
    """
    Creates a basic task with some metadata.

    :param verbose: Verbosity.
    :param working_dir: Working directory given by the user.
    :param task_class: Type of the Task.
    :param name: App name.
    :param seed: Random seed.
    :param version: App version.
    :return:
    """
    return task_class(
        creation_date=filename_friendly_date(),
        app_name=name,
        app_version=version,
        root_dir=working_dir,
        rng_seed=seed,
        verbose=verbose)


def parse_args() -> argparse.Namespace:
    """
    Parses commandline arguments with ``argparse``.

    :return: Parsed arguments.
    """
    top_parser = argparse.ArgumentParser(add_help=False)
    subparsers = top_parser.add_subparsers(dest='task')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "directory", type=str,
        help="Path to directory for saving and loading")

    labeled_dataset_parser = subparsers.add_parser(
        'labeled-dataset',
        description='Build a labeled dataset from an image and label map',
        help='Build a labeled dataset from an image and label map',
        parents=[parent_parser])
    labeled_dataset_parser.set_defaults(func=main)
    labeled_dataset_parser.add_argument(
        '--with-cvat', action="store_true", dest="with_cvat",
        help="build a labeled dataset by fragmenting a given image "
             "and labeling the tiles on https://cvat.ai")

    unlabeled_dataset_parser = subparsers.add_parser(
        'unlabeled-dataset',
        description='Build an unlabeled dataset from an image',
        help='Build an unlabeled dataset from an image',
        parents=[parent_parser])
    unlabeled_dataset_parser.set_defaults(func=main)

    train_parser = subparsers.add_parser(
        'train',
        description='Train a model with a dataset',
        help='Train a model with a dataset',
        parents=[parent_parser])
    train_parser.set_defaults(func=main)

    predict_parser = subparsers.add_parser(
        'predict',
        description='Make predictions with a model and dataset',
        help='Make predictions with a model and dataset',
        parents=[parent_parser])
    predict_parser.set_defaults(func=main)

    pipeline_parser = subparsers.add_parser(
        'pipeline',
        description='Run the full pipeline to create a labeled dataset, '
                    'train a model, and optionally make predictions '
                    'on a dataset at the end',
        help='Run the full pipeline to create a labeled dataset, '
             'train a model, and optionally make predictions '
             'on a dataset at the end',
        parents=[parent_parser])
    pipeline_parser.set_defaults(func=main)

    top_args, unknown = top_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description='Segmentation of arbitrary features '
                    'in very high resolution remote sensing imagery',
        parents=[top_parser])
    parser.add_argument(
        "--assume-default", "-d", action="store_true", dest='assume_default',
        help="assume the default option for all "
             "prompts and run non-interactively")

    args = parser.parse_args(unknown, namespace=top_args)

    if not vars(args).get('task'):
        parser.print_help()
        raise SystemExit(1)

    return args


def close() -> int:
    """
    Graceful application exit on ``KeyboardInterrupt``.

    return: Exit code 1.
    """
    print(flush=True)
    print("KeyboardInterrupt received, closing ...")
    return 1


if __name__ == '__main__':
    try:
        main()
        success("Task completed!")
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(close())
