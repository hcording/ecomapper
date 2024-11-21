"""
``Task`` abstract base class.
Tasks represent principal components of the EcoMapper pipeline.
"""

import importlib
import json
import os
import sys
from abc import ABC, ABCMeta, abstractmethod
from typing import Type, TypeVar

import jsonpickle

from ecomapper.utils.json_util import save_to_json, try_deserialize_json_file
from ecomapper.utils.path_util import expand_path, make_dir
from ecomapper.utils.print_util import err, warn


class SaveOnInitMeta(type):
    def __call__(cls, *args, **kwargs):
        """
        Method managing logic in and around ``__init__``.
        Tasks cannot be saved while they are being constructed.
        After construction, the ``_loading`` field is removed, to allow Tasks
        to be saved to disk.
        However, if the Task is in read-only mode, the field is kept to prevent
        saving.
        """
        instance = super().__call__(*args, **kwargs)
        if not kwargs.get('_read_only'):
            delattr(instance, '_loading')

        return instance


class CombinedMeta(ABCMeta, SaveOnInitMeta):
    pass


class Task(ABC, metaclass=CombinedMeta):
    """
    A ``Task`` is an abstraction for a principal component of the EcoMapper
    pipeline.

    The working directory of a Task is its ``root_dir``, where the Task itself
    and the files and directories it owns are stored.
    Except for other Tasks and third-party library sources, a Task
    should be fully self-contained in its ``root_dir`` to avoid file scatter
    and enable easy relocation to a different disk or machine.

    Tasks can be seen as configuration files with additional features for
    (de)serialization and maintenance of saved metadata.
    The storage format for serialized Tasks is JSON, with the noteworthy
    addition that duplicate keys are treated as error during (de)serialization.

    All Tasks have basic metadata, such as creation date, application name,
    and version.
    Subclasses of ``Task`` can maintain additional, task-specific metadata.

    Subclasses should implement the ``_get_input`` and ``_run`` methods to
    get required inputs from the user and execute the operation associated with
    the Task.

    The ``Task`` abstract base class additionally dictates the basic operations
    that are executed when the ``run`` method is called on a Task. These
    are:

    #. Checking if paths to external data have become invalid.
    #. Getting user input during Task creation.
    #. Executing the Task.

    Tasks can optionally maintain one or more ``Journal`` instances for
    tracking progress on specific operations and restoring the last stable
    state after an interruption.

    Notes
    -----
    Any field ending in "_ext" in ``Task`` and its subclasses is assumed to be
    an external path pointing to a location outside ``root_dir``.
    Such fields are set to ``None`` in ``_check_external_paths`` during
    Task loading if the external resource was moved or deleted.
    """

    T = TypeVar('T', bound='Task')

    TASK_FILENAME = "task.json"
    """
    Name of the file to which serialized tasks are written.
    Use the ``task_file`` property to get the absolute path to this file.
    """

    @property
    def task_file(self) -> str:
        """
        Absolute path to the Task file, e.g., "/home/tasks/mytask/task.json".

        :return: Task file path.
        """
        return os.path.join(self.root_dir, self.TASK_FILENAME)

    def __init__(self,
                 creation_date: str,
                 app_name: str,
                 app_version: str,
                 root_dir: str,
                 rng_seed: int,
                 verbose: bool,
                 **kwargs):
        self._loading = True

        self.creation_date = creation_date
        self.app_name = app_name
        self.app_version = app_version

        root_dir = expand_path(root_dir)
        if (not os.path.exists(root_dir)
                or not os.path.exists(
                    os.path.join(root_dir, Task.TASK_FILENAME))):
            try:
                make_dir(root_dir, require_empty=True)
            except RuntimeError:
                err("Cannot create a Task here, "
                    "directory exists and is not empty")
                sys.exit(1)

        self.root_dir = root_dir
        """
        The absolute working directory of this Task, where all its auxiliary
        files and directories should be stored.
        """

        self.rng_seed = rng_seed
        """
        The random seed for the application.
        Stored in each task to pass on to other libraries, e.g., to
        MMSegmentation when training a model with a ``TrainTask``.
        """

        # TODO replace with log level and use logger
        self.verbose = verbose

        self.journal_dir = make_dir(os.path.join(self.root_dir, 'journals'))
        """
        Directory for storing Journals to backup and restore progress.
        """

    def __delattr__(self, item) -> None:
        """
        Removes the named attribute ``key`` from ``self``, and calls
        ``self._try_save()`` afterward.

        :param item: Name of an attribute of ``self`` to remove.
        """
        super().__delattr__(item)
        self._try_save()

    def __setattr__(self, key, value) -> None:
        """
        Sets the attribute ``key`` of ``self`` to ``value``, and calls
        ``self._try_save()`` afterward.

        :param key: Name of an attribute of ``self``.
        :param value: Value to set.
        """
        super().__setattr__(key, value)
        self._try_save()

    @classmethod
    def load(cls: Type[T], path: str, require_done: bool = True,
                 verbose: bool = True,
                 read_only: bool = False) -> T:
        result = cls.try_load(path, require_done, verbose, read_only)
        if result is None:
            raise RuntimeError(f"Failed to load Task from '{path}'")
        return result

    @classmethod
    def try_load(cls: Type[T], path: str, require_done: bool = True,
                 verbose: bool = True,
                 read_only: bool = False) -> T | None:
        """
        Attempts to load a Task of type ``cls`` from ``path``.

        :param path: Directory containing a Task.
        :param require_done: Whether the Task has to be completed.
        :param verbose: TODO
        :param read_only: If ``True``, the Task will not be written to disk
            after being initialized.
        :return: ``None`` if loading failed, otherwise the loaded Task.
        """
        path = expand_path(path)
        if not os.path.exists(path):
            if verbose:
                warn("Path does not exist")
            return None

        task_file = os.path.join(path, Task.TASK_FILENAME)
        if not os.path.isfile(task_file):
            if verbose:
                warn("Directory does not contain a task")
            return None

        task_dict = try_deserialize_json_file(task_file, Task)
        if task_dict is None:
            # No warning here, ``try_deserialize_json_file``
            # will generate a warning if loading failed
            return None

        task_dict['_read_only'] = read_only

        # Update the ``root_dir`` before loading, so that ``save``
        # can run properly when called from ``__init__``
        task_dict['root_dir'] = path

        task: Task = jsonpickle.decode(json.dumps(task_dict))

        task_class: Type[Task] = Task._module_string_to_type(
            task_dict['py/object'])
        task: Task = task_class.from_dict(task.__dict__, read_only)

        if not isinstance(task, cls):
            if verbose:
                warn(f"Directory contains a {task.__class__.__name__}, "
                     f"but a {cls.__name__} is expected")
            return None

        if not task.is_done() and require_done:
            if verbose:
                warn(f"Directory contains a valid {task.__class__.__name__} "
                     f"but the task has not been completed yet")
            return None

        if require_done:
            task.run()

        return task

    @classmethod
    def from_dict(cls, task_dict, read_only=False):
        task_dict['_read_only'] = read_only
        return cls(**task_dict)

    def _try_save(self):
        """
        Writes this task to the ``self.task_file`` path on disk, if the
        ``root_dir`` is defined and the Task is not being loaded.
        """
        if (hasattr(self, 'root_dir')
                and self.root_dir is not None
                and not hasattr(self, '_loading')):
            save_to_json(self, self.task_file)

    def __str__(self) -> str:
        """
        String representation of this instance, as pretty-printed JSON
        string with indentation level 4.

        :return: ``self`` as ``str``.
        """
        jsonpickle = importlib.import_module('jsonpickle')
        return jsonpickle.encode(self, indent=4)

    def __repr__(self):
        return self.__str__()

    def _check_external_paths(self) -> None:
        """
        Sets fields of ``self`` ending in "_ext" to ``None`` if the value of
        the field is not a valid path.
        """
        for key, value in self.__dict__.items():
            if not value or not key.endswith('_ext'):
                continue

            path = expand_path(value)
            if not os.path.exists(path):
                warn(f"Path to {key} has become invalid: {path}")
                self.__setattr__(key, None)

    @abstractmethod
    def _get_input(self) -> None:
        """
        Reads required inputs for this Task from stdin.
        """

    @abstractmethod
    def _run(self) -> None:
        """
        Do not call this method on a Task directly, use ``run`` instead.

        Executes the operation associated with this Task.
        This method will be called after all necessary checks for the Task are
        completed.
        """

    @abstractmethod
    def is_done(self) -> bool:
        """
        Indicates Task completion.

        :return: ``True`` iff the Task is completed, ``False`` otherwise.
        """

    def run(self) -> None:
        """
        Checks the validity of task metadata, gets required inputs from the
        user, and runs the Task.
        """
        self._check_external_paths()
        self._get_input()
        self._run()

    def register_journal(self, name: str) -> str:
        Journal = importlib.import_module('ecomapper.core.journal').Journal

        return Journal(os.path.join(
            self.journal_dir, name)).journal_file

    @staticmethod
    def _module_string_to_type(module_string: str) -> Type['Task']:
        module_path, class_name = module_string.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
