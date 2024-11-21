"""
Creation of labeled datasets from images and label maps.
"""

import os
import shutil
import sys
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ecomapper.core.journal import Journal
from ecomapper.task.unlabeled_dataset_task import UnlabeledDatasetTask
from ecomapper.task.unlabeled_dataset_task import check_fragment
from ecomapper.utils.file_util import extract_folder_from_zip
from ecomapper.utils.geo_image_util import generate_distinct_colors
from ecomapper.utils.geo_image_util import get_channels
from ecomapper.utils.geo_image_util import get_dims
from ecomapper.utils.geo_image_util import get_metadata
from ecomapper.utils.geo_image_util import has_field
from ecomapper.utils.geo_image_util import is_same_crs
from ecomapper.utils.geo_image_util import reproject_from_raster
from ecomapper.utils.geo_image_util import try_get_file_type
from ecomapper.utils.geo_image_util import valid_georeference
from ecomapper.utils.path_util import expand_path
from ecomapper.utils.path_util import make_dir
from ecomapper.utils.print_util import clear
from ecomapper.utils.print_util import err
from ecomapper.utils.print_util import warn
from ecomapper.utils.prompt_util import prompt_with_predicates
from ecomapper.utils.tqdm_util import get_bar_format


class DatasetTask(UnlabeledDatasetTask):
    """
    Extension of ``UnlabeledDatasetTask`` to add labels to the dataset.
    """

    def __init__(self,
                 label_map_file_ext: str | None = None,
                 label_legend: dict[str, int] | None = None,
                 num_classes: int | None = None,
                 palette: list[list[int]] | None = None,
                 label_map_class_field: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.label_map_file_ext = label_map_file_ext
        """
        Path to a label map from which to create label tiles for this dataset.
        A label map is a single channel (grayscale), 8-bit
        image. The value of each pixel is a class ID corresponding to
        ``self.label_legend``.
        """

        self.label_legend = label_legend
        """
        Dictionary mapping class names to class IDs.
        """

        self.num_classes = num_classes
        """
        Number of classes in the given label_legend, and thus in the label map
        at ``self.label_map_file_ext``.
        """

        self.palette = palette
        """
        List of lists, where each sublist contains three integers for R, G, and
        B color values.
        """

        self.label_tiles_dir = make_dir(os.path.join(
            self.root_dir, 'label_tiles'))
        """
        Directory in which to store label tiles.
        """

        self.dataset_distribution_journal_file = \
            self.register_journal('dataset_distribution_journal')
        """
        Records which label tiles have already been considered in the
        dataset distribution statistic.
        """

        self.sample_weights_journal_file = \
            self.register_journal('sample_weights_journal')
        """
        Records which label tiles have already been considered in the
        sample weights computation.
        """

        self.palettize_journal_file = \
            self.register_journal('palettize_journal')
        """
        Records which labels have already been palettized.
        """

        self.label_map_class_field = label_map_class_field
        """
        Stores the name of the field specifying the class ID of each pixel.
        Only set if ``self.label_map_file_ext`` is a vector file.
        """

    def _get_input(self):
        self.tile_width, self.tile_height = super().get_tile_dims()
        super()._get_input()

        self.label_map_file_ext = self.get_label_map()

        # For vector label maps, the name of the field indicating classes is
        # required
        label_map_type = try_get_file_type(self.label_map_file_ext)
        if label_map_type == 'vector' and not self.label_map_class_field:
            self.label_map_class_field = prompt_with_predicates(
                "Label map is a vector file. "
                "Please provide the name of the field in the label map "
                "which specifies the pixel class (e.g., \"id\"):",
                [(lambda field: has_field(self.label_map_file_ext, field),
                  "Label map does not contain this field")])[0]

        if not self.label_legend:
            self.label_legend = DatasetTask.get_label_legend()

        if not self.num_classes:
            self.num_classes = len(self.label_legend)

        if not self.palette:
            self.palette = generate_distinct_colors(self.num_classes)

    def get_label_map(self) -> str:
        """
        Gets the label map, and checks its validity.

        :return: Filepath to label map.
        """
        label_map_file_ext = self.label_map_file_ext

        # The label map is required as long as fragmenting is not complete
        if not self.label_map_file_ext \
                and not super().is_done():
            # Get the label map and ensure it  has the right dimensions
            label_map_file_ext = prompt_with_predicates(
                "Path to label map:",
                [(lambda x: try_get_file_type(x) is not None, ""),
                 (lambda x: (try_get_file_type(x) == 'vector'
                             or get_channels(x) == 1),
                  "Label map cannot have more than 1 band/channel"),
                 (lambda x: (try_get_file_type(x) == 'vector'
                             or get_dims(x) == get_dims(self.image_file_ext)),
                  "Dimensions of orthomosaic differ from label map")])[0]

            # Vector labels cannot be rasterized without a frame of reference,
            # which is only provided by georeferenced image inputs.
            label_map_type = try_get_file_type(label_map_file_ext)
            if label_map_type == 'vector' \
                    and not valid_georeference(self.image_metadata[-2],
                                               self.image_metadata[-1]):
                err("Cannot use vector label map "
                    "when input image is not georeferenced")
                raise SystemExit(1)

            # If the label map is using a different coordinate reference
            # system, reproject it to match the CRS of the input image
            if (valid_georeference(self.image_metadata[-2],
                                   self.image_metadata[-1])
                    and not is_same_crs(self.image_file_ext,
                                        label_map_file_ext)):
                warn("The label map will be reprojected "
                     "to match the CRS of the input image")

                new_label_map_file = os.path.join(
                    self.root_dir,
                    os.path.basename(label_map_file_ext))

                reproject_from_raster(self.image_file_ext,
                                      label_map_file_ext,
                                      new_label_map_file)

                label_map_file_ext = new_label_map_file

            # Borrow the georeference information from the label map
            # if the image has no such information but the label map does
            if label_map_type == 'raster' and not valid_georeference(
                    self.image_metadata[-2], self.image_metadata[-1]):
                label_metadata = get_metadata(label_map_file_ext)
                if valid_georeference(label_metadata[-2], label_metadata[-1]):
                    print("Using georeference information from label map")
                    self.image_metadata = label_metadata

        return label_map_file_ext

    @staticmethod
    def get_label_legend() -> dict:
        """
        Gets the mapping of class names to class IDs from stdin.

        :return: Dictionary mapping class names to class IDs.
        """
        clear()
        print("Please now provide the mapping of pixel values "
              "in the label map to class names.\n"
              "Note that the value 0 is reserved for the background class.\n")

        classes = {"background": 0}
        cid = 1

        while True:
            print("Current summary:")
            for k, v in classes.items():
                print(f'{k}: {v}')
            print()

            # Get the next class name
            # Several predicates must be fulfilled for this name to be valid
            inp = prompt_with_predicates(
                "Enter the name of a class"
                + (" or press ENTER to finish"
                   if len(classes) > 1 else '') + (
                    ' (type u then ENTER to undo the last entry):' if len(
                        classes) > 1 else ':'),
                predicates=[(lambda x: len(x) > 0 or len(classes) > 1,
                             "At least one class other than "
                             "background is required"),
                            (lambda x: len(classes) > 1 or x != 'u',
                             "Nothing to undo"),
                            (lambda x: x.strip().lower() not in classes.keys(),
                             "Class name already in use"),
                            (lambda x: ' ' not in x.strip().lower(),
                             "Class name cannot contain spaces")])[
                0].strip().lower()

            if inp == 'u' and len(classes) > 1:
                classes.popitem()
                cid -= 1
                clear()
                continue

            if inp == '':
                break

            classes[inp] = cid
            cid += 1
            clear()

        return classes

    def is_dataset_distribution_done(self) -> bool:
        """
        Checks whether the dataset distribution has been calculated.

        :return: :code:`True` if the dataset distribution has been calculated,
            :code:`False` otherwise.
        """
        return Journal(self.dataset_distribution_journal_file).is_done(
            self.label_tiles_dir)

    def is_sample_weights_done(self) -> bool:
        """
        Checks whether the sample weights have been calculated.

        :return: :code:`True` if the sample weights have been calculated,
            :code:`False` otherwise.
        """
        return Journal(self.sample_weights_journal_file).is_done(
            self.label_tiles_dir)

    def is_palettize_done(self) -> bool:
        """
        Checks whether the labels have been palettized.

        :return: :code:`True` if labels have been palettized, :code:`False`
            otherwise.
        """
        return Journal(self.palettize_journal_file).is_done(
            self.label_tiles_dir)

    def is_done(self) -> bool:
        return (super().is_done()
                and self.label_legend is not None
                and self.num_classes is not None
                and self.palette is not None
                and self.is_dataset_distribution_done()
                and self.is_sample_weights_done()
                and self.is_palettize_done())

    def _run(self):
        build_dataset(self)

    @classmethod
    def with_cvat(cls, dataset_task: UnlabeledDatasetTask):
        """
        Creates a ``DatasetTask`` using CVAT labels in Cityscapes 1.0 format.

        :param dataset_task: Initial Task used for splitting the input image.

        :return: ``DatasetTask`` with labels from CVAT.
        """
        # Journal to record progress of this operation
        journal = Journal(os.path.join(dataset_task.journal_dir,
                                       'cityscapes_journal'))

        # A label legend is needed to make a labeled dataset
        if 'label_legend' not in journal:
            label_legend = DatasetTask.get_label_legend()
            journal['label_legend'] = label_legend
        label_legend = journal['label_legend']

        label_dir = make_dir(os.path.join(dataset_task.root_dir,
                                          'label_tiles'))

        # Extract the labels from the CVAT zip file, and journal the produced
        # directory path
        if 'gt_fine_dir' not in journal:
            cvat_zip = expand_path(prompt_with_predicates(
                "Path to zip file downloaded from CVAT:",
                [(lambda x: os.path.isfile(expand_path(x)),
                  "Given path does not point to a file")])[0])
            extract_folder_from_zip(cvat_zip, label_dir, 'gtFine')
            journal['gt_fine_dir'] = os.path.join(label_dir, 'gtFine')
        gt_fine_dir = journal['gt_fine_dir']

        # If this folder exists, we are not yer done moving label files into
        # the dataset
        if os.path.exists(gt_fine_dir):
            # Get files that are still here
            all_files = set(Path(gt_fine_dir).rglob('*.png'))

            # Get only the label files that use the class ID as pixel value
            label_files = set(
                (_ for _ in all_files if '_labelIds.png' in str(_)))

            # Move the relevant label tiles into the dataset, and
            # delete everything else
            for label_file in label_files:
                shutil.move(label_file, label_dir)
            shutil.rmtree(gt_fine_dir)

        # Get the final paths of all label tiles
        label_tiles = glob(os.path.join(label_dir, '*'))
        label_tile_stems = set((Path(_).stem for _ in label_tiles))

        # CVAT doesn't export labels that have only background (0) pixels.
        # This method will produce these tiles to fill in what's missing.
        def empty():
            return np.zeros((dataset_task.tile_height,
                             dataset_task.tile_width))

        # Note which tiles still need to be checked
        remaining_tiles = journal.get_remaining(
            dataset_task.image_tiles_dir)
        remaining_tiles = [Path(_).stem for _ in remaining_tiles]

        # The missing tiles are filled in with all 0s
        for image_tile in remaining_tiles:
            if (image_tile + '_gtFine_labelIds') not in label_tile_stems:
                Image.fromarray(empty(), mode='L').save(
                    os.path.join(label_dir, f'{image_tile}.png'))
            journal[image_tile] = None

        # Now the label tiles from CVAT need to be renamed to match the
        # image tiles
        len_prefix = len(Path(os.listdir(
            dataset_task.image_tiles_dir)[0]).stem)
        for label_tile in label_tiles:
            if '_labelIds.png' not in label_tile:
                continue

            shutil.move(label_tile, os.path.join(
                label_dir,
                Path(label_tile).stem[:len_prefix] + '.png'))

        # Finally transform the unlabeled dataset task
        # into a labeled dataset task
        DatasetTask(
            None,
            label_legend,
            len(label_legend),
            None,
            **dataset_task.__dict__
        ).run()

        # Clean up the journal
        os.remove(journal.journal_file)


def build_dataset(task: DatasetTask) -> None:
    """
    Entry point for creating a labeled dataset.

    :param task: Dataset Task.
    """
    check_fragment(task,
                   task.label_map_file_ext,
                   task.label_tiles_dir,
                   task.label_map_class_field)

    assert (len(os.listdir(task.image_tiles_dir)) ==
            len(os.listdir(task.label_tiles_dir))), \
        "Number of image tiles does not equal number of label tiles"

    class_occurrences, class_occurrences_per_sample = \
        check_calculate_dataset_distribution(task)
    check_sample_weights(task, class_occurrences, class_occurrences_per_sample)
    check_palettize_labels(task)


def check_calculate_dataset_distribution(task: DatasetTask) \
        -> tuple[list[int], list[list[int]]]:
    """
    Calculates the dataset distribution, if not yet computed.

    :param task: Dataset task.
    :return: A tuple of two lists: total occurrences per class and class
        occureneces per tile.
    """
    journal = Journal(task.dataset_distribution_journal_file)

    if not task.is_dataset_distribution_done():
        print()

        calculate_dataset_distribution(
            task.num_classes,
            journal.get_remaining(task.label_tiles_dir),
            journal)

    class_occurrences = np.sum(list(journal.values()), axis=0)
    assert min(class_occurrences) > 0, \
        f"All classes listed in the label legend must appear in the " \
        f"dataset at least once, but the following class did not: " \
        f"{np.argmin(class_occurrences)}"

    assert (sum(class_occurrences) ==
            len(journal) * task.tile_width
            * task.tile_height)

    return list(class_occurrences), list(journal.values())


def check_sample_weights(
        task: DatasetTask,
        class_occurrences: list[int],
        class_occurrences_per_tile: list[list[int]]) -> None:
    r"""
    Calculates the sample weights, if not yet computed.

    :param task: Dataset task.
    :param class_occurrences: A list of :math:`C` integers that each give the
        pixel-wise count of a class :math:`c \in C` over all labels in the
        training set, where :math:`C` is the number of classes.
    :param class_occurrences_per_tile: A list of :math:`N` sublists, where
        each sublist contains :math:`C` integers that each give the
        pixel-wise count of a class :math:`c \in C` over a label tile in the
        training set, where :math:`C` is the number of classes and :math:`N`
        the number of tiles in the training set.
    """
    journal = Journal(task.sample_weights_journal_file)
    if not task.is_sample_weights_done():
        print()

        calculate_sample_weights(
            journal.get_remaining(task.label_tiles_dir),
            class_occurrences,
            class_occurrences_per_tile,
            journal)


def check_palettize_labels(task: DatasetTask) -> None:
    """
    Palettizes the label tiles, if at least one tile is not yet palettized.

    :param task: Dataset task.
    """
    if not task.is_palettize_done():
        print()
        journal = Journal(task.palettize_journal_file)

        palettize_labels(journal.get_remaining(task.label_tiles_dir),
                         task.palette,
                         journal)


def calculate_dataset_distribution(
        num_classes: int,
        labels_to_calculate: list[str],
        dataset_distribution_journal: Journal) -> None:
    """
    Algorithm to compute the dataset distribution.

    :param num_classes: Number of classes in the dataset.
    :param labels_to_calculate: Labels to still process for the computation.
    :param dataset_distribution_journal: Journal to record progress.
    """

    num_processed = len(dataset_distribution_journal)
    for label in tqdm(labels_to_calculate,
                      desc="Calculating dataset distribution",
                      bar_format=get_bar_format(),
                      total=num_processed + len(labels_to_calculate),
                      initial=num_processed):
        m = np.array(Image.open(label).convert("L"), dtype=np.uint8)

        counts = np.bincount(m.ravel(), minlength=num_classes)
        if len(counts) != num_classes:
            err("Encountered a label tile with "
                "more classes than specified in the label legend, "
                "please double check the label "
                "legend and label map file.")
            err(f"Invalid label tile is here: {label}")
            sys.exit(1)

        dataset_distribution_journal[Path(label).stem] = \
            counts


def calculate_sample_weights(
        labels_to_calculate: list[str],
        class_occurrences: list[int],
        class_occurrences_per_tile: list[list[int]],
        sample_weights_journal: Journal | dict) -> None:
    r"""
    Caclulates a weight for every sample in the training dataset.
    The weight indicates the (unnormalized) probability of a sample being
    drawn into the next batch of training data.

    :param labels_to_calculate: Labels to calculate weights for,
        as a list of paths.
    :param class_occurrences: A list of :math:`C` integers that each give the
        pixel-wise count of a class :math:`c \in C` over all labels in the
        training set, where :math:`C` is the number of classes.
    :param class_occurrences_per_tile: A list of :math:`N` sublists, where
        each sublist contains :math:`C` integers that each give the
        pixel-wise count of a class :math:`c \in C` over a label tile in the
        training set, where :math:`C` is the number of classes and :math:`N`
        the number of tiles in the training set.
    :param sample_weights_journal: Records which sample weights have already
        been computed.

    Examples
    --------
    >>> journal = dict()
    >>> calculate_sample_weights(['./image_0.png', './image_1.png'],
    ...    [2, 16], [[0, 9], [2, 7]], journal)
    >>> journal['image_0'], journal['image_1']
    (0.5625, 1.4375)
    """
    num_processed = len(sample_weights_journal)
    for idx, label in tqdm(enumerate(labels_to_calculate),
                           desc="Calculating sample weights",
                           total=num_processed + len(labels_to_calculate),
                           bar_format=get_bar_format(),
                           initial=num_processed):
        relative_class_occurrences = \
            [x / class_occurrences[i] for i, x in
             enumerate(class_occurrences_per_tile[idx])]
        sample_weights_journal[Path(label).stem] = \
            sum(relative_class_occurrences)


def palettize_labels(labels_to_palettize: list[str],
                     palette: list[list[int]],
                     palettize_journal: Journal) -> None:
    """
    Palettizes labels, so that they can be passed to an MMSegmentation model
    for training.

    :param labels_to_palettize: Labels to palettize.
    :param palette: Color palette to use.
    :param palettize_journal: Records which labels have already been
        palettized.
    """
    num_processed = len(palettize_journal)
    for file in tqdm(labels_to_palettize,
                     desc="Palettizing labels",
                     total=num_processed + len(labels_to_palettize),
                     bar_format=get_bar_format(),
                     initial=num_processed):
        seg_img = Image.open(file).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(file)

        palettize_journal[Path(file).stem] = None
