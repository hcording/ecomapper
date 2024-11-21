"""
Utility for testing ``DatasetTask`` functions.
"""

import os

from ecomapper.core.journal import Journal
from tests.functional.utils.input_util import _here


def build_dataset_input_image_tif_file():
    return os.path.join(build_dataset_input_dir(), 'image.tif')


def build_dataset_input_image_jpg_file():
    return os.path.join(build_dataset_input_dir(), 'image.jpg')


def build_dataset_input_label_map_tif_file():
    return os.path.join(build_dataset_input_dir(),
                        'label_map.tif')


def build_dataset_input_label_map_geojson_file():
    return os.path.join(build_dataset_input_dir(),
                        'label_map.geojson')


def build_dataset_input_label_map_shp_file():
    return os.path.join(build_dataset_input_dir(),
                        'label_map.shp')


def build_dataset_input_label_map_tif_file_epsg_4326():
    return os.path.join(build_dataset_input_dir(),
                        'label_map_epsg_4326.tif')


def build_dataset_input_label_map_shp_file_epsg_4326():
    return os.path.join(build_dataset_input_dir(),
                        'label_map_epsg_4326.shp')


def build_dataset_input_label_map_geojson_file_epsg_4326():
    return os.path.join(build_dataset_input_dir(),
                        'label_map_epsg_4326.geojson')


def build_dataset_cvat_zip():
    return os.path.join(build_dataset_input_dir(), 'cvat.zip')


def build_dataset_input_dir():
    return os.path.join(_here().parent.parent,
                        'data', 'mock_build_dataset_inputs')


def simple_dataset_input():
    return [
        '', '',  # confirm 512x512 tiles
        build_dataset_input_image_tif_file(),  # input image?
        build_dataset_input_label_map_tif_file(),  # label map?
        'chayote', 'border', ''  # label legend
    ]


def simple_dataset_no_labels_input():
    return [
        '', '',  # confirm 512x512 tiles
        build_dataset_input_image_tif_file(),  # input image?
    ]


def cvat_input():
    return ['', '', build_dataset_input_image_tif_file(),
            '',  # any key to continue
            'chayote', '',
            build_dataset_cvat_zip()]


def remove_half_journal(directory, task_class, journal_field_name):
    task = task_class.try_load(directory, require_done=False)
    journal = Journal(task.__dict__[journal_field_name])
    keys = sorted(journal.keys()).copy()
    half = len(journal) // 2
    for key in keys[half:]:
        del journal[key]
