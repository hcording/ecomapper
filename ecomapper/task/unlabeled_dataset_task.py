"""
Creation of unlabeled datasets from images.
"""

import os

from ecomapper.core.fragment import fragment
from ecomapper.core.journal import Journal
from ecomapper.core.safe_context_manager import SafeContextManager
from ecomapper.task.task import Task
from ecomapper.utils.geo_image_util import get_grid_dims, get_metadata
from ecomapper.utils.path_util import make_dir
from ecomapper.utils.print_util import warn
from ecomapper.utils.prompt_util import prompt_image, prompt_integer


class UnlabeledDatasetTask(Task):

    @property
    def default_tile_width(self):
        """
        The default tile width for images to input to the model.
        """
        return 512

    @property
    def default_tile_height(self):
        """
        The default tile height for images to input to the model.
        """
        return 512

    def __init__(self,
                 image_file_ext: str | None = None,
                 image_metadata: list | None = None,
                 tile_width: int | None = None,
                 tile_height: int | None = None,
                 total_num_tiles: int | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.image_file_ext = image_file_ext
        """
        Path to an image from which to create tiles for the dataset.
        """

        self.tile_width = tile_width
        """
        Width of tiles in the dataset.
        """

        self.tile_height = tile_height
        """
        Height of tiles in the dataset.
        """

        self.image_tiles_dir = make_dir(
            os.path.join(self.root_dir, 'image_tiles'))
        """
        Directory in which to store image tiles.
        """

        self.image_metadata = image_metadata
        """
        Metadata describing the image at ``self.image_file_ext``.
        Includes width, height, and georeference information (if present).
        """

        self.fragment_journal_file = \
            self.register_journal("fragment_journal")
        """
        Records which parts of the image have already been split into tiles.
        """

        self.total_num_tiles = total_num_tiles
        """
        The total number of tiles in the input image, before filtering out any
        tiles.
        """

    def is_done(self) -> bool:
        return (self.total_num_tiles is not None
                and self.tile_width is not None
                and self.tile_height is not None
                and self.image_metadata is not None
                and self.is_fragment_done())

    def is_fragment_done(self) -> bool:
        """
        Checks whether the input file at ``self.image_file_ext`` has been
        fragmented into tiles.

        :return: ``True`` if all tiles were created, ``False`` otherwise.
        """
        if (not self.image_metadata
                or not self.tile_width
                or not self.tile_height
                or not self.total_num_tiles):
            return False

        fragment_journal = Journal(self.fragment_journal_file)
        return len(fragment_journal) == self.total_num_tiles

    def _get_input(self):
        self.tile_width, self.tile_height = self.get_tile_dims()

        # Need the external image file until the Task is done
        if not self.image_file_ext and not self.is_done():
            self.image_file_ext = prompt_image(
                "Path to orthomosaic (input image):")

        if not self.image_metadata:
            self.image_metadata = get_metadata(self.image_file_ext)

    def get_tile_dims(self) -> tuple[int, int]:
        """
        Gets the tile dimensions for splitting input data from stdin.

        :return: Tuple of width, height.
        """
        if not self.tile_width or not self.tile_height:
            print("Please specify the tile size for input splitting:")
            tile_width = prompt_integer(
                "Tile width:", 4, None,
                self.default_tile_width)
            tile_height = prompt_integer(
                "Tile height:", 4, None,
                self.default_tile_height)
            return tile_width, tile_height  # pytype: disable=bad-return-type
        return self.tile_width, self.tile_height

    def _run(self):
        check_fragment(self)

    @staticmethod
    def compare_tile_size(dataset_a, dataset_b):
        """
        Given two dataset, checks whether their tile dimensions are identical.

        :param dataset_a: First dataset.
        :param dataset_b: Second dataset.
        :return: ``True`` if tile dimensions match between datasets, ``False``
            otherwise.
        """
        tile_size_a = (dataset_a.tile_width,
                       dataset_a.tile_height)
        tile_size_b = (dataset_b.tile_width,
                       dataset_b.tile_height)
        match = tile_size_a == tile_size_b
        if not match:
            warn("Given dataset is incompatible: Expected "
                 f"tiles of {tile_size_a[0]}x"
                 f"{tile_size_a[1]}, but the given dataset "
                 f"uses tiles of {tile_size_b[0]}x"
                 f"{tile_size_b[1]}")

        return match


def check_fragment(dataset_task: UnlabeledDatasetTask,
                   label_map_file_ext: str | None = None,
                   label_tiles_dir: str | None = None,
                   label_map_class_field: str | None = None) -> None:
    """
    Fragments the input image, if not yet fragmented.

    :param dataset_task: Dataset task.
    :param label_map_file_ext: Filepath to label map.
    :param label_tiles_dir: Path to directory for storing label tiles
        (optional). Provided by ``DatasetTask`` when fragmenting image and
        label map.
    :param label_map_class_field: Name of field containing classes in label map
        (optional). Only required when label map is a vector file.
    """
    if not dataset_task.is_fragment_done():
        print()
        fragment_journal = Journal(dataset_task.fragment_journal_file)

        width, height = dataset_task.image_metadata[:2]
        num_cols, num_rows = get_grid_dims(
            width, height, dataset_task.tile_width, dataset_task.tile_height)
        dataset_task.total_num_tiles = num_cols * num_rows

        # Wrapping the fragmentation in a ``SafeContextManager`` ensures
        # that progress recorded on each core is merged into a Journal on the
        # main thread before the application closes.
        with SafeContextManager(
                cleanup=lambda x: Journal.merge_journals(
                    dataset_task.journal_dir, fragment_journal)):
            fragment(dataset_task.image_file_ext,
                     label_map_file_ext,
                     label_map_class_field,
                     dataset_task.journal_dir,
                     dataset_task.tile_width,
                     dataset_task.tile_height,
                     num_rows,
                     num_cols,
                     dataset_task.image_tiles_dir,
                     label_tiles_dir,
                     fragment_journal)
