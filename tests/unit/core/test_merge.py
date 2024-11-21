"""
Unit tests for merging predictions.
"""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from ecomapper.core.merge import merge_predictions
from ecomapper.core.merge import write_segmentation_map
from ecomapper.utils.geo_image_util import get_grid_dims
from ecomapper.utils.numpy_util import join_splits_horizontal
from ecomapper.utils.numpy_util import join_splits_vertical


class TestMerge(unittest.TestCase):

    @staticmethod
    def make_overlap_tiles(grid_top_left, grid_top_right, grid_bottom_left,
                           grid_bottom_right):
        grid_top_center = join_splits_horizontal(grid_top_left, grid_top_right)
        grid_bottom_center = join_splits_horizontal(grid_bottom_left,
                                                    grid_bottom_right)
        grid_middle_left = join_splits_vertical(grid_top_left,
                                                grid_bottom_left)
        grid_middle_center = join_splits_vertical(grid_top_center,
                                                  grid_bottom_center)
        grid_middle_right = join_splits_vertical(grid_top_right,
                                                 grid_bottom_right)
        return (grid_top_center, grid_middle_left, grid_middle_center,
                grid_middle_right, grid_bottom_center)

    def run_test(self, expected, tiles, grid_w, grid_h, tile_w, tile_h,
                 assertions_enabled=True):
        with tempfile.TemporaryDirectory() as temp_dir:
            padding = len(str(len(tiles)))
            for i, t in enumerate(tiles):
                Image.fromarray(t).convert("L").save(
                    os.path.join(
                        temp_dir, f'image_{str(i).zfill(padding)}.png'))

            merged_path = os.path.join(temp_dir, 'merged.dat')
            metadata = [grid_w, grid_h, (0., 1., 0., 0., 0., 1.), '']
            merge_predictions(
                merged_path,
                temp_dir,
                False,
                {},  # type: ignore
                [(tile_w, tile_h) for _ in range(len(tiles))],
                metadata,
                tile_w,
                tile_h)

            write_segmentation_map(merged_path,
                                   os.path.join(temp_dir, 'merged.tif'),
                                   metadata)

            if assertions_enabled:
                result = np.memmap(merged_path,
                                   dtype=np.uint8,
                                   shape=(grid_h, grid_w),
                                   mode='r+')

                self.assertEqual(expected.shape, result.shape)
                self.assertTrue(np.all(expected == result))

    def test_merge_1(self):
        grid_top_left = np.arange(1, 31, dtype=np.uint8).reshape(5, 6)
        grid_top_right = np.arange(61, 91, dtype=np.uint8).reshape(5, 6)

        grid_bottom_left = np.arange(61, 91, dtype=np.uint8).reshape(5, 6)
        grid_bottom_right = np.arange(1, 31, dtype=np.uint8).reshape(5, 6)

        (grid_top_center, grid_middle_left, grid_middle_center,
         grid_middle_right, grid_bottom_center) = self.make_overlap_tiles(
            grid_top_left, grid_top_right, grid_bottom_left, grid_bottom_right)

        tiles = [grid_top_left, grid_top_center, grid_top_right,
                 grid_middle_left, grid_middle_center, grid_middle_right,
                 grid_bottom_left, grid_bottom_center, grid_bottom_right]

        expected = np.vstack((
            np.hstack((grid_top_left, grid_top_right)),
            np.hstack((grid_bottom_left, grid_bottom_right))
        ))

        self.run_test(expected, tiles,
                      grid_w=12, grid_h=10, tile_w=6, tile_h=5)

    def test_merge_2(self):
        grid_top_left = np.arange(0, 25, dtype=np.uint8).reshape(5, 5)
        grid_top_right = np.arange(50, 75, dtype=np.uint8).reshape(5, 5)

        grid_bottom_left = np.arange(100, 125, dtype=np.uint8).reshape(5, 5)
        grid_bottom_right = np.arange(125, 150, dtype=np.uint8).reshape(5, 5)

        (grid_top_center, grid_middle_left, grid_middle_center,
         grid_middle_right, grid_bottom_center) = self.make_overlap_tiles(
            grid_top_left, grid_top_right, grid_bottom_left, grid_bottom_right)

        tiles = [grid_top_left, grid_top_center, grid_top_right,
                 grid_middle_left, grid_middle_center, grid_middle_right,
                 grid_bottom_left, grid_bottom_center, grid_bottom_right]

        expected = np.vstack((
            np.hstack((grid_top_left, grid_top_right)),
            np.hstack((grid_bottom_left, grid_bottom_right))
        ))

        self.run_test(expected, tiles, 10, 10, 5, 5)

    def test_merge_3(self):
        grid_top_left = np.arange(0, 16, dtype=np.uint8).reshape(4, 4)
        grid_top_right = np.arange(32, 48, dtype=np.uint8).reshape(4, 4)

        grid_bottom_left = np.arange(100, 116, dtype=np.uint8).reshape(4, 4)
        grid_bottom_right = np.arange(132, 148, dtype=np.uint8).reshape(4, 4)

        (grid_top_center, grid_middle_left, grid_middle_center,
         grid_middle_right, grid_bottom_center) = self.make_overlap_tiles(
            grid_top_left, grid_top_right, grid_bottom_left, grid_bottom_right)

        tiles = [grid_top_left, grid_top_center, grid_top_right,
                 grid_middle_left, grid_middle_center, grid_middle_right,
                 grid_bottom_left, grid_bottom_center, grid_bottom_right]

        expected = np.vstack((
            np.hstack((grid_top_left, grid_top_right)),
            np.hstack((grid_bottom_left, grid_bottom_right))
        ))

        self.run_test(expected, tiles, 8, 8, 4, 4)

    def test_merge_4(self):
        grid_top_left = np.arange(0, 16, dtype=np.uint8).reshape(4, 4)
        grid_top_right = np.arange(32, 48, dtype=np.uint8).reshape(4, 4)

        grid_lower_left = np.arange(56, 72, dtype=np.uint8).reshape(4, 4)
        grid_lower_right = np.arange(72, 88, dtype=np.uint8).reshape(4, 4)

        a, b, c, d, e = self.make_overlap_tiles(grid_top_left,
                                                grid_top_right,
                                                grid_lower_left,
                                                grid_lower_right)

        grid_bottom_left = np.arange(0, 16, dtype=np.uint8).reshape(4, 4)
        grid_bottom_right = np.arange(32, 48, dtype=np.uint8).reshape(4, 4)

        e2, f, g, h, i = self.make_overlap_tiles(grid_lower_left,
                                                 grid_lower_right,
                                                 grid_bottom_left,
                                                 grid_bottom_right)

        self.assertListEqual(e.tolist(), e2.tolist())

        tiles = [
            grid_top_left, a, grid_top_right,
            b, c, d,
            grid_lower_left, e, grid_lower_right,
            f, g, h,
            grid_bottom_left, i, grid_bottom_right
        ]

        expected = np.vstack((
            np.hstack((grid_top_left, grid_top_right)),
            np.hstack((grid_lower_left, grid_lower_right)),
            np.hstack((grid_bottom_left, grid_bottom_right))
        ))

        self.run_test(expected, tiles, 8, 12, 4, 4)

    def test_big(self):
        # 10_000 x 12_000 image (W x H)
        img_w = 10_000
        img_h = 12_000
        tile_w = 512
        tile_h = 512
        grid_w, grid_h = get_grid_dims(img_w, img_h, tile_w, tile_h)
        tiles = []
        flag = False
        for y in range(grid_w):
            for x in range(grid_h):
                if flag:
                    tiles.append(np.zeros((tile_w, tile_h), dtype=np.uint8))
                else:
                    tiles.append(np.ones((tile_w, tile_h), dtype=np.uint8))
                flag = not flag

        tiles = np.array(tiles, dtype=np.uint8)
        self.run_test(None, tiles, img_w, img_h, tile_w, tile_h,
                      assertions_enabled=False)
