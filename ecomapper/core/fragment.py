"""
Functions for fragmenting images and label maps into tiles.
"""

import multiprocessing
import os
import time
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.managers import DictProxy
from multiprocessing.managers import SyncManager

import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from osgeo.gdal import Dataset
from tqdm import tqdm

from ecomapper.core.journal import Journal
from ecomapper.utils.geo_image_util import extract_label_map_region
from ecomapper.utils.tqdm_util import get_bar_format


def fragment_slice(image_file: str,
                   image_metadata: list,
                   label_map_file: str | None,
                   label_map_class_field: str | None,
                   y_start: int,
                   y_end: int,
                   num_tiles: int,
                   tiles_per_row: int,
                   image_tiles_dir: str,
                   label_tiles_dir: str,
                   tile_width: int, tile_height: int,
                   pid: int,
                   fragment_journal_shared: DictProxy,
                   journal_dir: str,
                   shared_counter,
                   lock) -> None:
    """
    Fragments a given range of rows from an input image into tiles with 50%
    horizontal and vertical overlap.

    :param image_file: Input image to fragment.
    :param image_metadata: List containing width, height, and geotransform of
        the input image.
    :param label_map_file: Label map file to fragment (optional).
    :param label_map_class_field: Name of field containing classes in label map
        (optional). Only required when label map is a vector file.
    :param y_start: Row to start fragmenting from (inclusive).
    :param y_end: Row to stop fragmenting at (exclusive).
    :param num_tiles: Number of tiles in the full grid, i.e., after all rows
        have been fragmented.
    :param tiles_per_row: Tiles per row, considering tiles have 50% overlap.
    :param image_tiles_dir: Path to directory to save image fragments to.
    :param label_tiles_dir: Path to directory to save label map fragments to.
    :param tile_width: Width of fragmented tiles. Tiles at the image boundary
        will be padded to reach this width.
    :param tile_height: Height of fragmented tiles. Tiles at the image boundary
        will be padded to reach this height.
    :param pid: Process ID, needed for splitting the image with more than 1
        process.
    :param fragment_journal_shared: The fragment journal of the previous run.
        Tiles already recorded in this journal are not fragmented again.
    :param journal_dir: Path to directory to save journal of this process to.
        The process journal is different from ``fragment_journal_shared``.
        After all processes finish, the journals are merged into
        ``fragment_journal_shared``.
    :param shared_counter: Non-threadsafe counter shared across processes, to
        inform main thread of fragmentation progress. Used for updating
        progress bar.
    :param lock: Threadsafe lock, used for updating ``shared_counter`` safely.
    """
    width, height, geo_transform = image_metadata

    # Horizontal and vertical ground sampling distance
    x_res = geo_transform[1]
    y_res = -geo_transform[5]

    # Tiles use the naming scheme "image_<number>.png".
    # <number> must be padded so that it has the same length for all tiles and
    # sorts correctly.
    padding = len(str(num_tiles))

    # Processes record their progress
    process_journal = Journal(os.path.join(
        journal_dir,
        f'fragment_journal_{pid}.joblib'))

    for y in range(y_start, y_end):
        for x in range(tiles_per_row):
            count = y * tiles_per_row + x

            tile_stem = f"image_{str(count).zfill(padding)}"

            # This tile was fragmented in a previous run
            if tile_stem in fragment_journal_shared:
                continue

            top = y * (tile_height // 2)
            left = x * (tile_width // 2)
            src_tile_width = min(tile_width, width - left)
            src_tile_height = min(tile_height, height - top)

            # Pull the tile under the current grid position into memory
            image_output_path = os.path.join(
                image_tiles_dir, tile_stem + '.jpg')
            image_tile = gdal.Translate("", image_file,
                                        srcWin=[left, top, src_tile_width,
                                                src_tile_height],
                                        outputType=gdal.GDT_Byte,
                                        format='MEM',
                                        noData=0)
            image_data = np.array(image_tile.ReadAsArray()).astype(np.uint8)

            # Discard alpha channels and multispectral data
            if image_data.shape[0] > 3:
                image_data = image_data[:3, :, :]
            mean = image_data.mean()

            # Filter out tiles that have no information
            if 0 <= mean < 1 or 254 < mean <= 255:
                process_journal[tile_stem] = None
                with lock:
                    shared_counter.value += 1
                continue

            # Pad images at the grid boundary with 0s to reach full size
            row_padding = 0
            col_padding = 0
            if image_data.shape[1] < tile_height \
                    or image_data.shape[2] < tile_width:
                row_padding = tile_height - image_data.shape[1]
                col_padding = tile_width - image_data.shape[2]

                image_data = np.pad(
                    image_data, ((0, 0), (0, row_padding), (0, col_padding)),
                    mode='constant', constant_values=0)

            # Save the image tile
            image_tile = gdal_array.OpenArray(image_data)
            gdal.Translate(image_output_path, image_tile,
                           format="JPEG",
                           creationOptions=["QUALITY=90"])

            # Repeat the process for the label tile, if a label map was given
            if label_map_file:
                label_output_path = os.path.join(
                    label_tiles_dir, tile_stem + '.png')

                # This function call also handles rasterization in case a
                # vector file was provided
                label_data = extract_label_map_region(
                    label_map_file, left,
                    top, src_tile_width,
                    src_tile_height, x_res,
                    y_res, geo_transform,
                    label_map_class_field)

                # Pad label tiles as well
                if row_padding > 0 or col_padding > 0:
                    label_data = np.pad(
                        label_data, ((0, row_padding), (0, col_padding)),
                        mode='constant', constant_values=0)

                # Save the label tile
                label_tile = gdal_array.OpenArray(label_data)
                gdal.Translate(label_output_path, label_tile,
                               format="PNG",
                               creationOptions=["ZLEVEL=6"])

            # Record progress
            process_journal[tile_stem] = (src_tile_width, src_tile_height)
            with lock:
                shared_counter.value += 1


def fragment(image_file: str,
             label_map_file: str | None,
             label_map_class_field: str | None,
             journal_dir: str,
             tile_width: int,
             tile_height: int,
             num_rows: int,
             num_cols: int,
             image_tiles_dir: str,
             label_tiles_dir: str | None,
             fragment_journal: Journal) -> None:
    """
    Fragments an input image into tiles with 50% horizontal and vertical
    overlap.

    :param num_cols: Number of columns in grid with 50% tile overlap.
    :param num_rows: Number of rows in grid with 50% tile overlap.
    :param fragment_journal: Journal to record fragmentation progress.
    :param image_file: Input image to fragment.
    :param label_map_file: Label map file to fragment (optional).
    :param label_map_class_field: Name of field containing classes in label map
        (optional). Only required when label map is a vector file.
    :param image_tiles_dir: Path to directory to save image fragments to.
    :param label_tiles_dir: Path to directory to save label map fragments to.
    :param tile_width: Width of fragmented tiles. Tiles at the image boundary
        will be padded to reach this width.
    :param tile_height: Height of fragmented tiles. Tiles at the image boundary
        will be padded to reach this height.
    :param journal_dir: Path to directory to save journals to.
    """
    # Split image rows of CPU cores
    y_ranges = calculate_row_ranges(num_rows)

    # Extract image metadata
    orthomosaic: Dataset = gdal.Open(image_file)
    width = orthomosaic.RasterXSize
    height = orthomosaic.RasterYSize
    gt = orthomosaic.GetGeoTransform()
    image_metadata = [width, height, gt]

    # Create a multiprocessing workflow
    manager: SyncManager
    with multiprocessing.Manager() as manager:
        shared_counter = manager.Value(int, len(fragment_journal))
        fragment_journal_shared = manager.dict(dict(fragment_journal))
        lock = manager.Lock()
        total = num_rows * num_cols

        # Generate futures over the computed row ranges.
        # The ``ProcessPoolExecutor`` will aim to use as many processes
        # as cores in the system.
        # Since row ranges are chunked, cores can pick up new row ranges
        # from the list if they finish early.
        futures: list[Future] = []
        with ProcessPoolExecutor() as executor:
            for pid, (y_start, y_end) in enumerate(y_ranges):
                futures.append(executor.submit(
                    fragment_slice,
                    image_file,
                    image_metadata,
                    label_map_file,
                    label_map_class_field,
                    y_start,
                    y_end,
                    total,
                    num_cols,
                    image_tiles_dir,
                    label_tiles_dir,
                    tile_width, tile_height,
                    pid,
                    fragment_journal_shared,
                    journal_dir,
                    shared_counter, lock))

            # Create a progress bar based on the shared counter
            with tqdm(total=total,
                      desc="Splitting data into tiles",
                      bar_format=get_bar_format(),
                      initial=len(fragment_journal)) as pbar:
                while True:
                    pbar.n = shared_counter.value
                    pbar.refresh()

                    if pbar.n >= total:
                        break

                    time.sleep(1)
                pbar.n = shared_counter.value
                pbar.refresh()

            for i, future in enumerate(futures):
                future.result()


def calculate_row_ranges(
        num_rows: int,
        num_processes: int | None = None,
        max_rows_per_range: int | None = None) -> list[tuple[int, int]]:
    """
    Splits a given number of rows into ranges of rows.
    If the rows cannot be evenly divided into ranges, an additional
    range is appended with the remainder.

    :param num_rows: Number of rows to split.
    :param num_processes: Number of processes to use. If not
        provided, set to number of cores in the system.
    :param max_rows_per_range: Maximum number of rows each range can hold.
    :return: List of ranges as (start (inclusive), end (exclusive)).

    Examples
    --------
    >>> calculate_row_ranges(400, num_processes=4, max_rows_per_range=512)
    [(0, 100), (100, 200), (200, 300), (300, 400)]

    >>> calculate_row_ranges(415, num_processes=4, max_rows_per_range=512)
    [(0, 103), (103, 206), (206, 309), (309, 412), (412, 415)]

    >>> calculate_row_ranges(101, num_processes=4, max_rows_per_range=15)
    [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 101)]

    >>> calculate_row_ranges(101, num_processes=1)
    [(0, 101)]

    >>> calculate_row_ranges(2, num_processes=8)
    [(0, 1), (1, 2)]
    """
    num_processes = multiprocessing.cpu_count() \
        if num_processes is None else num_processes

    if num_processes == 1:
        return [(0, num_rows)]

    if num_rows <= num_processes:
        num_processes = num_rows

    if max_rows_per_range is None:
        max_rows_per_range = max(1, num_rows // num_processes)

    rows_per_process = max(1,
                           min(max_rows_per_range, num_rows // num_processes))

    num_ranges = num_rows // rows_per_process
    y_ranges = [(i * rows_per_process, (i + 1) * rows_per_process) for i in
                range(num_ranges)]

    remaining_rows = num_rows % rows_per_process
    if remaining_rows != 0:
        y_ranges.append((num_ranges * rows_per_process, num_rows))

    return y_ranges
