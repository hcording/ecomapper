"""
Unit tests for image fragmentation.
"""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from ecomapper.core.fragment import fragment
from ecomapper.core.journal import Journal
from ecomapper.utils.geo_image_util import get_grid_dims
from ecomapper.utils.geo_image_util import get_metadata
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_image_tif_file
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_label_map_tif_file


class TestFragment(unittest.TestCase):
    def test_simple(self):
        self.run_test(512, 512)

    def test_rectangular_tiles_1(self):
        self.run_test(512, 1024)

    def test_rectangular_tiles_2(self):
        self.run_test(1024, 512)

    def test_strange_tiles(self):
        self.run_test(111, 110)

    def setUp(self) -> None:
        self.fragment_journal = Journal('./fj.joblib')

    def tearDown(self) -> None:
        os.remove(self.fragment_journal.journal_file)

    def run_test(self, tile_w, tile_h):
        image = build_dataset_input_image_tif_file()
        metadata = get_metadata(image)
        cols, rows = get_grid_dims(metadata[0], metadata[1], tile_w, tile_h)
        with tempfile.TemporaryDirectory() as temp_dir:
            fragment(image,
                     build_dataset_input_label_map_tif_file(),
                     None,
                     temp_dir,
                     tile_w,
                     tile_h,
                     rows,
                     cols,
                     temp_dir,
                     temp_dir,
                     self.fragment_journal)

            self.fragment_journal = Journal.merge_journals(
                temp_dir, self.fragment_journal)

            # Check that image and label tiles are correct
            self.check_tiles(temp_dir, tile_w, tile_h, cols, rows,
                             self.fragment_journal, extension='jpg',
                             channels=3)
            self.check_tiles(temp_dir, tile_w, tile_h, cols, rows,
                             self.fragment_journal, extension='png',
                             channels=1)

    def check_tiles(self, directory, tile_w, tile_h, cols, rows,
                    fragment_journal, extension, channels):
        count = 0
        for image in fragment_journal:
            self.assertEqual(
                f'image_{str(count).zfill(len(str(cols * rows)))}', image)
            if fragment_journal[image] is not None:
                x = np.array(Image.open(os.path.join(directory,
                                                     image + f'.{extension}')),
                             dtype=np.uint8)
                self.assertEqual(tile_w, x.shape[1])
                self.assertEqual(tile_h, x.shape[0])
                if len(x.shape) == 3:
                    self.assertEqual(channels, x.shape[2])
                else:
                    self.assertEqual(1, channels)
            count += 1
