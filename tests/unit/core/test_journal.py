"""
Unit tests for journaling.
"""

import os.path
import tempfile
import unittest

from ecomapper.core.journal import Journal


class TestJournal(unittest.TestCase):
    def test_ctor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_file = os.path.join(temp_dir, 'journal.joblib')
            j = Journal(journal_file)
            self.assertTrue(os.path.exists(journal_file))
            self.assertEqual(0, len(j))
            self.assertEqual(journal_file, j.journal_file)

    def test_ctor_append_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_file = os.path.join(temp_dir, 'journal')
            expected_path = os.path.join(temp_dir, 'journal.joblib')
            j = Journal(journal_file)
            self.assertTrue(os.path.exists(expected_path))
            self.assertEqual(0, len(j))
            self.assertEqual(expected_path, j.journal_file)

    def test_ctor_existing_journal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_file = os.path.join(temp_dir, 'journal.joblib')
            j = Journal(journal_file)
            j['entry'] = 999
            j = Journal(journal_file)
            self.assertTrue(os.path.exists(journal_file))
            self.assertEqual(1, len(j))
            self.assertEqual(journal_file, j.journal_file)

    def test_setitem_saves_to_disk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_file = os.path.join(temp_dir, 'journal.joblib')
            j = Journal(journal_file)
            j['entry'] = 999
            j = Journal(journal_file)
            self.assertEqual(1, len(j))
            self.assertTrue('entry' in j)
            self.assertEqual(999, j['entry'])

    def test_delitem_saves_to_disk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_file = os.path.join(temp_dir, 'journal.joblib')
            j = Journal(journal_file)
            j['entry'] = 999
            j['entry2'] = 1000
            j = Journal(journal_file)
            del j['entry']
            j = Journal(journal_file)
            self.assertEqual(1, len(j))
            self.assertFalse('entry' in j)
            self.assertTrue('entry2' in j)
            self.assertEqual(1000, j['entry2'])

    def test_from_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            d = {'entry': 999}
            journal_file = os.path.join(temp_dir, 'journal.joblib')
            j = Journal.from_dict(d, journal_file)
            self.assertTrue(os.path.exists(journal_file))
            self.assertEqual(1, len(j))
            self.assertTrue('entry' in j)
            self.assertEqual(999, j['entry'])

    def test_merge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_file_1 = os.path.join(temp_dir, 'journal1.joblib')
            journal_file_2 = os.path.join(temp_dir, 'journal2.joblib')
            journal_file_3 = os.path.join(temp_dir, 'journal3.joblib')

            j1 = Journal.from_dict({'entry1': 999}, journal_file_1)
            Journal.from_dict({'entry2': 1000}, journal_file_2)
            Journal.from_dict({'entry3': 1001}, journal_file_3)
            expected = Journal.from_dict(
                {'entry1': 999, 'entry2': 1000, 'entry3': 1001},
                journal_file_1)

            j1 = Journal.merge_journals(temp_dir, j1)
            self.assertDictEqual(expected, j1)

    def test_get_remaining(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_file = os.path.join(temp_dir, 'journal.joblib')
            j = Journal.from_dict({'file_1': None, 'file_2': None},
                                  journal_file)
            file_dir = os.path.join(temp_dir, 'files')
            os.mkdir(file_dir)
            to_do = ['file_3.txt', 'file_4.txt']
            expected = []
            for t in to_do:
                with open(os.path.join(file_dir, t), 'w') as f:
                    f.write('...')
                    expected.append(f.name)

            actual = j.get_remaining(file_dir)
            self.assertListEqual(expected, actual)
