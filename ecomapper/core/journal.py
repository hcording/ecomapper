"""
Journal class for tracking progress of various Tasks.
"""

import os
from glob import glob
from pathlib import Path

from ecomapper.utils.joblib_util import joblib_dump_atomic
from ecomapper.utils.joblib_util import joblib_load_typed


class Journal(dict):
    """
    The ``Journal`` class is an extension of Python's builtin ``dict``.
    Journals allow atomic saving to disk when items are added/removed, and add
    functionality to calculate the remainder of items to process based on
    the Journal content and a given directory with files to produce.
    """

    def __init__(self, journal_file: str):
        super().__init__()
        assert not os.path.isdir(journal_file)
        if not journal_file.endswith('.joblib'):
            journal_file += '.joblib'
        self.journal_file = journal_file
        if not os.path.exists(self.journal_file):
            self.create_empty()
        else:
            x = self._load_journal_as_dict()
            self.update(x)

    def __delitem__(self, key) -> None:
        """
        Deletes the (key, value) entry mapped by ``key`` and writes the Journal
        to disk.

        :param key: Key to remove entry with.
        """
        super().__delitem__(key)
        self.dump()

    def __setitem__(self, key, value) -> None:
        """
        Sets ``self[key] = value`` and writes the Journal to disk.

        :param key: Key to insert.
        :param value: Value to insert.
        """
        super().__setitem__(key, value)
        self.dump()

    def _load_journal_as_dict(self) -> dict:
        """
        Loads a Journal saved as joblib file from disk as ``dict`` instance.

        :return: Journal as dictionary.
        """
        return joblib_load_typed(self.journal_file, dict)

    def get_remaining(self, directory: str) -> list[str]:
        """
        Calculates items missing from the Journal based on the items in the
        given directory.

        :param directory: Directory to scan.
        :return: List of filepaths to process,
            sorted in ascending order by name.
        """
        existing = sorted(glob(os.path.join(directory, '*')))
        existing = {Path(x).stem: x for x in existing}
        remaining = sorted(
            set(existing.keys()).difference(set(self.keys())))
        return [existing[filename] for filename in remaining]

    def create_empty(self) -> None:
        """
        Writes an empty Journal to ``self.journal_file``.
        """
        self.clear()
        self.dump()

    def dump(self) -> None:
        """
        Writes the Journal to ``self.journal_file``.
        """
        joblib_dump_atomic(dict(self), self.journal_file)

    @classmethod
    def from_dict(cls, dictionary: dict, journal_file: str) -> 'Journal':
        """
        Instantiates a journal from a dictionary.

        :param dictionary: Dictionary to get data from.
        :param journal_file: Filepath for new Journal.
        :return: Created Journal.
        """
        journal = cls(journal_file)
        journal.update(dictionary)
        journal.dump()
        return journal

    @staticmethod
    def merge_journals(journal_dir, journal) -> 'Journal':
        """
        Merges the journals in ``journal_dir`` into the given ``journal``.

        :param journal_dir: Path to directory containing journals.
        :param journal: Journal to merge other journals into.
        """
        # Find journals
        process_journals = glob(os.path.join(
            journal_dir,
            '*.joblib'))

        # Merge them with the given journal
        for process_journal in process_journals:
            journal |= Journal(process_journal)

        # Ensure the entries are sorted and write changes to disk
        result = Journal.from_dict(dict(sorted(journal.items())),
                                   journal.journal_file)
        return result

    def is_done(self, label_tiles_dir) -> bool:
        """
        Checks whether all items for this Journal have been completed.

        :param label_tiles_dir: Directory to check for items and compare with
            journal entries.
        :return: ``True`` if no items remain for this Journal, ``False``
            otherwise.
        """
        remaining_labels = self.get_remaining(label_tiles_dir)
        return len(remaining_labels) == 0
