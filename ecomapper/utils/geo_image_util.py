"""
Manipulation of georeferenced raster and vector data.
"""

import colorsys
import math
import os

import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo.gdal import Dataset
from osgeo.gdal import osr
from osgeo.ogr import DataSource
from osgeo.ogr import Driver
from osgeo.ogr import Layer
from osgeo.osr import CoordinateTransformation
from osgeo.osr import SpatialReference

from ecomapper.utils.print_util import warn

gdal.UseExceptions()


def try_load_image(path: str) -> gdal.Dataset | None:
    """
    Attempts to open a given image.

    :return: An ``osgeo,gdal.Dataset`` if the image could be opened, otherwise
        ``None``.
    """
    if path == '':
        return None

    try:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
    except RuntimeError:
        warn("Given path does not point to a valid image")
        return None

    return dataset


def get_dims(path: str) -> tuple[int, int]:
    """
    Returns the width and height of an image.

    :param path: Filepath to an image.
    :return: Tuple of width, height.
    """
    dataset: gdal.Dataset = try_load_image(path)
    w, h = dataset.RasterXSize, dataset.RasterYSize
    del dataset
    return w, h


def get_channels(path: str) -> int:
    """
    Returns the number of channels (i.e., bands) in the given image.

    :param path: Filepath to an image.
    :return: Number of channels.
    """
    dataset: gdal.Dataset = gdal.Open(path)
    raster_count = dataset.RasterCount
    del dataset
    return raster_count


def generate_distinct_colors(n: int) -> list[list[int]]:
    """
    Generates :code:`n` visually distinct colors in RGB format.

    :param n: Number of colors to generate.
    :return: List of lists, where each sublist
        contains three integers for R, G, and B color values.

    Examples
    --------
    >>> generate_distinct_colors(3)
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    """
    # Keep saturation and value constant, and
    # sample hue `n` times at equal intervals
    colors_in_hsv = [(i / n, 1, 1) for i in range(n)]

    colors_in_rgb = [
        [int(val * 255) for val in colorsys.hsv_to_rgb(*color)] for
        color in colors_in_hsv]
    return colors_in_rgb


def get_grid_dims(
        width: int, height: int,
        tile_width: int, tile_height: int) -> tuple[int, int]:
    """
    Calculates the number of columns and rows for a grid obtained by splitting
    an image of ``width`` * ``height`` into tiles of ``tile_width`` *
    ``tile_height``, where tiles overlap by 50%.

    :param width: Image width.
    :param height: Image height.
    :param tile_width: Length of tiles along x-axis.
    :param tile_height: Length of tiles along y-axis.
    :return: Tuple of ``num_cols, num_rows``.

    Examples
    --------
    >>> get_grid_dims(768, 512, 128, 128)
    (11, 7)
    """
    num_cols = 1 if width < tile_width \
        else math.ceil((width - tile_width) / (tile_width / 2) + 1)
    num_rows = 1 if height < tile_height \
        else math.ceil((height - tile_height) / (tile_height / 2) + 1)
    return num_cols, num_rows


def valid_georeference(
        geo_transform: tuple[float, float, float, float, float, float],
        projection: str) -> bool:
    """
    Checks whether the given geotransform and projection form a valid
    georeference.

    :param geo_transform: Affine transformation.
    :param projection: Coordinate reference system (CRS).
    :return: ``True`` if the data are a valid georeference, ``False``
        otherwise.
    """
    return geo_transform != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0) \
        and projection != ''


def get_metadata(image_file) -> list:
    """
    Return metadata for the given image.

    :param image_file: Path to an image.
    :return: Tuple of (width, height, channels, geo_transform, projection).
        The last two items are ``(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)`` and ``''``
        (empty string) if the image is not georeferenced.
    """
    dataset: Dataset = gdal.Open(image_file)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    channels = dataset.RasterCount
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    if not valid_georeference(geo_transform, projection):
        warn(f"Image is not georeferenced: {image_file}")

    del dataset
    return [width, height, channels, geo_transform, projection]


def try_get_file_type(file_path: str) -> str | None:
    """
    Check the type of file using GDAL/OGR.

    :param file_path: Path to the file.
    :return: 'raster' for raster files, 'vector' for vector files, or 'unknown'
        otherwise.
    """
    # Rasters can be opened with GDAL
    try:
        dataset = gdal.Open(file_path)
        if dataset:
            del dataset
            return "raster"
    except RuntimeError:
        try:
            # Vectors can be opened with OGR
            dataset = ogr.Open(file_path)
            if dataset:
                del dataset
                return "vector"
        except RuntimeError:
            return None


def has_field(vector_file: str, field_name: str) -> bool:
    """
    Checks whether a vector file has a field with the given name.

    :param vector_file: Path to the vector file.
    :param field_name: Name of the field to check for.
    :return: ``True`` if the field exists, ``False`` otherwise.
    """
    dataset: DataSource = ogr.Open(vector_file)
    if not dataset:
        raise ValueError(f"Failed to open {vector_file}. "
                         f"Ensure it is a valid vector file.")

    layer = dataset.GetLayer(0)
    for i in range(layer.GetLayerDefn().GetFieldCount()):
        if layer.GetLayerDefn().GetFieldDefn(i).GetName() == field_name:
            del layer
            del dataset
            return True

    del layer
    del dataset
    return False


def compute_subregion_extents(left: int, top: int, width: int, height: int,
                              geo_transform: tuple[
                                  int, int, int, int, int, int]) -> list[int]:
    """
    Converts a subregion (bounding box) to georeferenced extents using a
    provided geo transform.

    :param left: X coordinate of left box points.
    :param top: Y coordinate of upper box points.
    :param width: Width of box.
    :param height: Height of box.
    :param geo_transform: Affine transformation coefficients.
    :return: Box extents as [min_x, min_y, max_x, max_y]
    """
    min_x = geo_transform[0] + left * geo_transform[1]
    max_x = geo_transform[0] + (left + width) * geo_transform[1]
    min_y = geo_transform[3] + (top + height) * geo_transform[5]
    max_y = geo_transform[3] + top * geo_transform[5]

    return [min_x, min_y, max_x, max_y]


def extract_label_map_region(
        label_map_file: str, left: int, top: int,
        width: int, height: int, x_res: float,
        y_res: float, geo_transform: tuple[int, int, int, int, int, int],
        field_name: str | None = None) -> np.ndarray:
    """
    Rasterizes a region (bounding box) of a vector file.

    :param label_map_file: Path to a label map.
    :param left: X coordinate of left box points.
    :param top: Y coordinate of upper box points.
    :param width: Width of box.
    :param height: Height of box.
    :param x_res: Horizontal resolution of rasterized output in georeferenced
        units (e.g., cm/px).
    :param y_res: Vertical resolution of rasterized output in georeferenced
        units (e.g., cm/px).
    :param geo_transform: Affine transformation coefficients.
    :param field_name: Name of field whose values to use for rasterization
        (e.g., "id"). Only required if given label legend is a vector file.
    :return: Numpy array containing rasterized region.
    """
    file_type = try_get_file_type(label_map_file)

    if file_type == 'raster':
        label_tile = gdal.Translate("", label_map_file,
                                    srcWin=[left, top, width, height],
                                    outputType=gdal.GDT_Byte,
                                    format="MEM",
                                    noData=0)
        if label_tile is not None:
            return np.array(label_tile.ReadAsArray()).astype(np.uint8)
        else:
            raise RuntimeError("Error while processing raster label map")

    elif file_type == 'vector':
        extents = compute_subregion_extents(
            left, top, width, height, geo_transform)
        options = gdal.RasterizeOptions(
            format="MEM",
            outputType=gdal.GDT_Byte,
            noData=0,
            xRes=x_res,
            yRes=y_res,
            outputBounds=extents,
            attribute=field_name)
        rasterized = gdal.Rasterize("", label_map_file, options=options)
        if rasterized is not None:
            return np.array(rasterized.ReadAsArray()).astype(np.uint8)
        raise ValueError("Error while processing vector label map")
    else:
        raise ValueError("Cannot process label map, unknown format")


def is_same_crs(raster_path: str, vector_path: str) -> bool:
    """
    Compare the Coordinate Reference System (CRS) of a raster and a vector
    file.

    :param raster_path: Path to raster file.
    :param vector_path: Path to vector file.
    :return: ``True`` if both CRS are the same, ``False`` otherwise.
    """
    raster_srs = get_coordinate_reference_system(raster_path)
    vector_srs = get_coordinate_reference_system(vector_path)
    return raster_srs.IsSame(vector_srs)


def get_coordinate_reference_system(file_path: str) -> SpatialReference:
    """
    Returns the CRS of the given image.

    :param file_path: Filepath to an image.
    :return: Image CRS.
    """
    file_type = try_get_file_type(file_path)

    if file_type == 'vector':
        vector_dataset: DataSource = ogr.Open(file_path)
        vector_layer: Layer = vector_dataset.GetLayer()
        crs: SpatialReference = vector_layer.GetSpatialRef()

    elif file_type == 'raster':
        raster_dataset: Dataset = gdal.Open(file_path)
        raster_wkt = raster_dataset.GetProjection()
        crs: SpatialReference = ogr.osr.SpatialReference()
        crs.ImportFromWkt(raster_wkt)

    else:
        raise RuntimeError("Cannot get CRS, unknown file format")

    return crs


def reproject_from_raster(raster_file: str, to_reproject_file: str,
                          output_file: str,
                          interpolation=gdal.GRA_NearestNeighbour) -> None:
    """
    Reproject a file to match the CRS of a raster file.

    :param raster_file: Path to a raster file.
    :param to_reproject_file: Path to a file (either a raster or vector file)
    :param output_file: Path to save the reprojected file to (if
        this file exists it will be overwritten).
    :param interpolation: Interpolation to use for reprojection. Should always
        be ``gdal.GRA_NearestNeighbour`` when reprojecting a label map.
    """
    file_type = try_get_file_type(to_reproject_file)

    if file_type == 'vector':
        reproject_vector_to_raster_crs(raster_file, to_reproject_file,
                                       output_file)
    elif file_type == 'raster':
        reproject_raster(raster_file, to_reproject_file,
                         output_file, interpolation)
    else:
        raise RuntimeError("Unknown file type for label map")


def reproject_vector_to_raster_crs(raster_file: str, vector_file: str,
                                   output_vector_file: str) -> None:
    """
    Reproject a vector file to match the CRS of a raster file.

    :param raster_file: Path to a raster file.
    :param vector_file: Path to a vector file.
    :param output_vector_file: Path to save the reprojected vector file to (if
        this file exists it will be overwritten).
    """
    # Get CRS from raster
    raster_dataset: Dataset = gdal.Open(raster_file)
    raster_wkt = raster_dataset.GetProjection()
    raster_srs: SpatialReference = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)

    # Get CRS from vector
    vector_dataset: DataSource = ogr.Open(vector_file)
    vector_layer: Layer = vector_dataset.GetLayer()
    vector_srs: SpatialReference = vector_layer.GetSpatialRef()

    # Create Coordinate Transformation
    # This will be used for the reprojection
    transform: CoordinateTransformation = \
        osr.CoordinateTransformation(vector_srs, raster_srs)

    # Create new reprojected dataset
    driver: Driver = vector_dataset.GetDriver()
    if os.path.exists(output_vector_file):
        os.remove(output_vector_file)

    output_dataset = driver.CreateDataSource(output_vector_file)
    output_layer = output_dataset.CreateLayer(
        vector_layer.GetName(),
        geom_type=vector_layer.GetGeomType(),
        srs=raster_srs)

    # Add attributes to the new dataset from the source vector file
    output_layer.CreateFields(vector_layer.schema)

    # Transform each feature and write it to the output layer
    for feature in vector_layer:
        geom = feature.GetGeometryRef()
        geom.Transform(transform)
        output_feature = ogr.Feature(output_layer.GetLayerDefn())
        output_feature.SetGeometry(geom)
        for i in range(feature.GetFieldCount()):
            output_feature.SetField(i, feature.GetField(i))
        output_layer.CreateFeature(output_feature)
        del output_feature

    del vector_dataset
    del output_dataset


def reproject_raster(source_file, to_reproject_file, output_raster,
                     interpolation=gdal.GRA_NearestNeighbour) -> None:
    """
    Reprojects ``input_raster`` to match the CRS of ``target_raster``, saving
    the result to ``output_raster``.

    :param source_file: Path to raster whose CRS to use for reprojection.
    :param to_reproject_file: Path to raster to reproject,
    :param output_raster: Path to save the reprojected raster.
    :param interpolation: Interpolation to use for reprojection. Should always
        be ``gdal.GRA_NearestNeighbour`` when reprojecting a label map.
    """
    # Get needed data from target
    target_ds = gdal.Open(source_file, gdal.GA_ReadOnly)
    target_proj = target_ds.GetProjection()
    target_geotrans = target_ds.GetGeoTransform()
    target_band = target_ds.GetRasterBand(1)
    x_res, y_res = target_band.XSize, target_band.YSize

    # Open input to be reprojected
    input_ds = gdal.Open(to_reproject_file, gdal.GA_ReadOnly)

    # Make an output image
    driver: gdal.Driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
    output_ds.SetGeoTransform(target_geotrans)
    output_ds.SetProjection(target_proj)

    # Reproject input with the desired CRS to the output
    gdal.ReprojectImage(input_ds, output_ds, input_ds.GetProjection(),
                        target_proj, interpolation)

    del output_ds
    del input_ds
    del target_ds
