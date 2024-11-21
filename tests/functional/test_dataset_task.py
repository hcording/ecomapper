"""
Functional tests for ``DatasetTask`` and ``UnlabeledDatasetTask``.
"""

import tempfile
import unittest
from unittest.mock import patch

from ecomapper.task.dataset_task import DatasetTask
from ecomapper.task.unlabeled_dataset_task import UnlabeledDatasetTask
from tests.functional.utils.input_util import run_with_inputs
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_image_jpg_file
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_image_tif_file
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_label_map_geojson_file
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_label_map_geojson_file_epsg_4326
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_label_map_shp_file
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_label_map_shp_file_epsg_4326
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_label_map_tif_file
from tests.functional.utils.test_dataset_task_util import \
    build_dataset_input_label_map_tif_file_epsg_4326
from tests.functional.utils.test_dataset_task_util import cvat_input
from tests.functional.utils.test_dataset_task_util import remove_half_journal
from tests.functional.utils.test_dataset_task_util import simple_dataset_input
from tests.functional.utils.test_dataset_task_util import \
    simple_dataset_no_labels_input


class TestDatasetTask(unittest.TestCase):

    def setUp(self) -> None:
        self.patchers: list = []

    def tearDown(self) -> None:
        for p in self.patchers:
            p.stop()

    @staticmethod
    def test_build_new_dataset_tif_label_map():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())

    @staticmethod
    def test_build_new_dataset_jpg_image():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, [
                    '', '',  # confirm 512x512 tiles
                    build_dataset_input_image_jpg_file(),  # input image?
                    build_dataset_input_label_map_tif_file(),  # label map?
                    'chayote', 'border', ''  # label legend
                ])

    @staticmethod
    def test_build_new_dataset_geojson_label_map():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, [
                    '', '',  # confirm 512x512 tiles
                    build_dataset_input_image_tif_file(),  # input image?
                    build_dataset_input_label_map_geojson_file(),  # label map?
                    'id',  # field for pixel IDs in vector file?
                    'chayote', 'border', ''  # label legend
                ])

    @staticmethod
    def test_build_new_dataset_shapefile_label_map():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, [
                    '', '',  # confirm 512x512 tiles
                    build_dataset_input_image_tif_file(),  # input image?
                    build_dataset_input_label_map_shp_file(),  # label map?
                    'id',  # field for pixel IDs in vector file?
                    'chayote', 'border', ''  # label legend
                ])

    def test_build_new_dataset_jpg_image_shapefile_label_map_raises_error(
            self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SystemExit) as cm:
                run_with_inputs(
                    'labeled-dataset',
                    temp_dir, [
                        '', '',  # confirm 512x512 tiles
                        build_dataset_input_image_jpg_file(),  # input image?
                        build_dataset_input_label_map_shp_file(),  # label map?
                        'id',  # field for pixel IDs in vector file?
                        'chayote', 'border', ''  # label legend
                    ])

            self.assertEqual(cm.exception.code, 1)

    @staticmethod
    def test_build_new_dataset_tif_label_map_reproject():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, [
                    '', '',  # confirm 512x512 tiles
                    build_dataset_input_image_tif_file(),  # input image?
                    build_dataset_input_label_map_tif_file_epsg_4326(),
                    # label map?
                    'chayote', 'border', ''  # label legend
                ])

    @staticmethod
    def test_build_new_dataset_shapefile_label_map_reproject():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, [
                    '', '',  # confirm 512x512 tiles
                    build_dataset_input_image_tif_file(),  # input image?
                    build_dataset_input_label_map_shp_file_epsg_4326(),
                    # label map?
                    'id',  # field for pixel IDs in vector file?
                    'chayote', 'border', ''  # label legend
                ])

    @staticmethod
    def test_build_new_dataset_geojson_label_map_reproject():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, [
                    '', '',  # confirm 512x512 tiles
                    build_dataset_input_image_tif_file(),  # input image?
                    build_dataset_input_label_map_geojson_file_epsg_4326(),
                    # label map?
                    'id',  # field for pixel IDs in vector file?
                    'chayote', 'border', ''  # label legend
                ])

    @staticmethod
    def test_build_new_dataset_no_labels():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'unlabeled-dataset',
                temp_dir, simple_dataset_no_labels_input())

    @staticmethod
    def test_existing_dataset():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs('labeled-dataset',
                            temp_dir, simple_dataset_input())
            run_with_inputs('labeled-dataset',
                            temp_dir, simple_dataset_input())

    @staticmethod
    def test_existing_dataset_no_labels():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'unlabeled-dataset',
                temp_dir, simple_dataset_no_labels_input())
            run_with_inputs(
                'unlabeled-dataset',
                temp_dir, simple_dataset_no_labels_input())

    def test_build_dataset_after_fragment_interrupt(self):
        p1 = patch('ecomapper.task.dataset_task'
                   '.check_calculate_dataset_distribution',
                   return_value=(None, None))
        p2 = patch('ecomapper.task.dataset_task'
                   '.check_sample_weights')
        p3 = patch('ecomapper.task.dataset_task'
                   '.check_palettize_labels')
        self.patchers += [p1, p2, p3]
        p1.start()
        p2.start()
        p3.start()

        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())
            remove_half_journal(
                temp_dir, DatasetTask, 'fragment_journal_file')

            p1.stop()
            p2.stop()
            p3.stop()

            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())

    def test_build_dataset_after_dataset_distribution_interrupt(self):
        p1 = patch('ecomapper.task.dataset_task'
                   '.check_sample_weights')
        p2 = patch('ecomapper.task.dataset_task'
                   '.check_palettize_labels')
        self.patchers += [p1, p2]
        p1.start()
        p2.start()

        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())
            remove_half_journal(
                temp_dir, DatasetTask,
                'dataset_distribution_journal_file')

            p1.stop()
            p2.stop()

            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())

    def test_build_dataset_after_sample_weights_interrupt(self):
        p1 = patch('ecomapper.task.dataset_task'
                   '.check_palettize_labels')
        self.patchers.append(p1)
        p1.start()

        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())
            remove_half_journal(
                temp_dir, DatasetTask,
                'sample_weights_journal_file')

            p1.stop()

            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())

    @staticmethod
    def test_build_dataset_after_palettize_interrupt():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())
            remove_half_journal(
                temp_dir, DatasetTask,
                'palettize_journal_file')

            run_with_inputs(
                'labeled-dataset',
                temp_dir, simple_dataset_input())

    @staticmethod
    def test_build_dataset_after_fragment_interrupt_no_labels():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'unlabeled-dataset',
                temp_dir, simple_dataset_no_labels_input())
            remove_half_journal(
                temp_dir, UnlabeledDatasetTask,
                'fragment_journal_file')
            run_with_inputs(
                'unlabeled-dataset',
                temp_dir, simple_dataset_no_labels_input())

    @staticmethod
    def test_build_dataset_from_cvat():
        with tempfile.TemporaryDirectory() as temp_dir:
            run_with_inputs(
                'labeled-dataset',
                temp_dir, cvat_input(),
                ['--with-cvat'])
