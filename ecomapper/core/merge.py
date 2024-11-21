"""
Functions for merging model predictions into a segmentation map.
"""

import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
from osgeo import gdal
from osgeo.gdal import Driver
from tqdm import tqdm

from ecomapper.core.journal import Journal
from ecomapper.utils.geo_image_util import get_grid_dims
from ecomapper.utils.geo_image_util import valid_georeference
from ecomapper.utils.tqdm_util import get_bar_format


def merge_predictions(merged_predictions_file: str,
                      prediction_tiles_dir: str,
                      is_using_scale: bool,
                      merge_journal: Journal,
                      original_dimensions: list[tuple[int, int]],
                      image_metadata,
                      tile_width,
                      tile_height) -> None:
    """
    Merges model predictions into a single grid by cropping overlapping tiles.

    :param merged_predictions_file: Filepath to save raw result to. This is not
        the final image, but it contains all image data to create a TIF from.
    :param prediction_tiles_dir: Path to directory containing model
        predictions.
    :param is_using_scale: Whether the model was trained with the ``SCALE`` and
        ``SCALE_METHOD`` environment variables. This must be known to rescale
        tiles during merging.
    :param merge_journal: Journal for saving merge progress.
    :param original_dimensions: List of tuples with as many items as tiles
        were generated from the input image (the image which the model made
        predictions on). Each item in the list is either a tuple of (width,
        height), indicating the tile size, or ``None``, indicating that the
        tile had a mean value close to 0 or 255 and was thus filtered out
        during splitting.
    :param image_metadata: Metadata of the input image, see
        ``ecomapper.utils.geo_image_util.get_metadata``.
    :param tile_width: Width of prediction tiles. All prediction tiles must
        have the same width.
    :param tile_height: Height of prediction tiles. All prediction tiles must
        have the same height.
    """
    image_width, image_height = image_metadata[:2]

    # The expected dimensions are used to rescale predictions from a model
    # using the ``SCALE`` and ``SCALE_METHOD`` environment variables.
    # This ensures that predictions fit the dimensions of the input image,
    # regardless of model input size.
    expected_tile_width, expected_tile_height = (tile_width, tile_height)
    expected_shape = (expected_tile_height, expected_tile_width)

    # Grid axes lengths
    num_tiles_x, num_tiles_y = get_grid_dims(
        image_width, image_height,
        expected_tile_width,
        expected_tile_height)

    # Create the output grid using memory mapping, since it wouldn't usually
    # fit into RAM
    if not os.path.exists(merged_predictions_file):
        merged_predictions = np.memmap(merged_predictions_file,
                                       dtype=np.uint8,
                                       mode='w+',
                                       shape=(image_height, image_width))
        merged_predictions.fill(0)
        merged_predictions.flush()
    else:
        merged_predictions = np.memmap(merged_predictions_file,
                                       dtype=np.uint8,
                                       mode='r+',  # Note the r+ instead of w+
                                       shape=(image_height, image_width))

    pred_tiles = sorted(glob(os.path.join(
        prediction_tiles_dir, '*')))

    # Restore progress, if any
    y, pred_count, grid_count = merge_journal.get('iter', (0, 0, 0))

    assert len(original_dimensions) == num_tiles_y * num_tiles_x, \
        "Insufficient number of tiles for grid of this size"

    if expected_tile_width < 4 or expected_tile_height < 4:
        raise ValueError("Tiles must be at least 4x4")

    if image_width < 8 or image_height < 8:
        raise ValueError("Image must be at least 8x8")

    # The amount to crop tiles by.
    # E.g., the left and right sides of a tile are each cropped by crop_x.
    # Sides touching the grid boundary are not cropped.
    crop_x, crop_y = expected_tile_width // 4, expected_tile_height // 4

    stride_x = expected_tile_width // 2 + crop_x
    stride_y = expected_tile_height // 2 + crop_y

    is_last_y = 1 + (grid_count // num_tiles_x) == num_tiles_y

    with tqdm(total=num_tiles_x * num_tiles_y,
              bar_format=get_bar_format(),
              desc="Merging predictions into segmentation map",
              initial=grid_count) as pbar:
        while y < image_height and not is_last_y:
            # Flush and save every row.
            # Saving at every tile would lead to excessive I/O use.
            merged_predictions.flush()
            merge_journal['iter'] = (y, pred_count, grid_count)

            is_last_y = 1 + (grid_count // num_tiles_x) == num_tiles_y
            is_last_x = (grid_count + 1) % num_tiles_x == 0

            x = 0
            while x < image_width and not is_last_x:
                original_tile_dimensions = original_dimensions[grid_count]

                # How big was this tile when splitting the input image?
                expected_tile_width = (
                    original_tile_dimensions[0]
                    if original_tile_dimensions is not None
                    else expected_shape[0])
                expected_tile_height = (
                    original_tile_dimensions[1]
                    if original_tile_dimensions is not None
                    else expected_shape[1])

                is_last_x = (grid_count + 1) % num_tiles_x == 0

                grid_step_y = stride_y
                grid_step_x = stride_x

                if is_last_x:
                    grid_step_x = image_width - x

                if is_last_y:
                    grid_step_y = image_height - y

                tile_y = 0
                tile_x = 0

                if y != 0:
                    tile_y += crop_y
                    if not is_last_y and (expected_tile_height % 2 == 0):
                        grid_step_y -= crop_y
                if is_last_y:
                    tile_y += expected_tile_height - (
                            image_height - y + tile_y)
                if x != 0:
                    tile_x += crop_x
                    if not is_last_x and (expected_tile_width % 2 == 0):
                        grid_step_x -= crop_x
                if is_last_x:
                    tile_x += expected_tile_width - (image_width - x + tile_x)

                # This image tile was filtered out, so no prediction exists.
                # Since the output grid was initialized with 0, we can just
                # skip this tile and the corresponding labels will be 0
                # (background).
                if original_tile_dimensions is None:
                    pbar.update(1)
                    grid_count += 1
                    x += grid_step_x

                    merge_journal['iter'] = (
                        y, merge_journal['iter'][1], grid_count)

                    continue

                # Load the prediction tile
                pred_tile = np.array(
                    Image.open(pred_tiles[pred_count]).convert("L"),
                    dtype=np.uint8)

                # Ensure the dimensions match the expected shape, or rescale
                # if the model used scaled inputs
                if not is_using_scale:
                    assert pred_tile.shape == expected_shape
                if pred_tile.shape != expected_shape:
                    pred_tile = cv2.resize(
                        pred_tile,
                        dsize=(expected_shape[1], expected_shape[0]),
                        interpolation=cv2.INTER_NEAREST)

                # Write the cropped tile to the grid
                merged_predictions[y:y + grid_step_y, x:x + grid_step_x] = \
                    pred_tile[
                    tile_y:tile_y + grid_step_y,
                    tile_x:tile_x + grid_step_x]

                x += grid_step_x
                pred_count += 1
                grid_count += 1
                pbar.update(1)

            y += grid_step_y  # noqa

    # Final flush and save
    merged_predictions.flush()
    merge_journal['iter'] = (y, merge_journal['iter'][1], grid_count)


def write_segmentation_map(merged_predictions_file: str,
                           output_file: str,
                           image_metadata: list) -> None:
    """
    Writes the raw data of the merged grid to disk as a TIF file.
    Georeference information is restored here if the input image contained
    any.

    :param merged_predictions_file: Path to merged predictions file.
    :param output_file: Filepath to save the segmentation map to.
    :param image_metadata: Metadata of the input image.
    """
    merged_predictions = np.memmap(
        merged_predictions_file,
        np.uint8,
        shape=(image_metadata[1], image_metadata[0]),
        mode='r+')

    driver: Driver = gdal.GetDriverByName('GTiff')
    new_dataset: gdal.Dataset = driver.Create(
        output_file,
        merged_predictions.shape[1], merged_predictions.shape[0],
        1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "ZLEVEL=6", "PREDICTOR=2"])

    new_dataset.GetRasterBand(1).WriteArray(merged_predictions)
    new_dataset.FlushCache()

    if valid_georeference(image_metadata[2], image_metadata[3]):
        new_dataset.SetGeoTransform(image_metadata[3])
        new_dataset.SetProjection(image_metadata[4])

    del new_dataset
    del driver
