"""
Registration of custom MMSegmentation components.
"""

import os
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Sized

import albumentations as A
import torch
from albumentations import BasicTransform
from cv2 import INTER_AREA
from cv2 import INTER_CUBIC
from mmcv import BaseTransform
from mmengine import DATASETS
from mmengine import DATA_SAMPLERS
from mmengine import HOOKS
from mmengine import TRANSFORMS
from mmengine.dataset import InfiniteSampler
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmseg.datasets import BaseSegDataset

from ecomapper.core.journal import Journal
from ecomapper.utils.joblib_util import joblib_load_typed


def register_sampler(sample_weights: list) -> None:
    """
    Registers the ``WeightedInfiniteSampler`` to MMSegmentation, to
    over-sample minority classes during training.

    :param sample_weights: Weights to bias sampling.
    """

    @DATA_SAMPLERS.register_module()
    class WeightedInfiniteSampler(InfiniteSampler):
        """
        Weighted sampling using a multinomial distribution.
        """

        def __init__(self,
                     dataset: Sized,
                     shuffle: bool = True,
                     seed: Optional[int] = None) -> None:
            super().__init__(dataset=dataset, shuffle=shuffle, seed=seed)

            self.weights = sample_weights
            """
            Weights to bias the sampling process.
            """

        def _infinite_indices(self) -> Iterator[int]:
            """
            Generates indices to tell the PyTorch dataloader which samples
            to include in each batch.

            :return: Integer iterator over the indices.
            """
            g = torch.Generator()
            g.manual_seed(self.seed)
            while True:
                if self.shuffle:
                    # Weighted sampling
                    yield from torch.multinomial(torch.tensor(self.weights),
                                                 self.size,
                                                 replacement=True,
                                                 generator=g).tolist()
                else:
                    yield from torch.arange(self.size).tolist()


def get_train_augmentations(tile_width: int,
                            tile_height: int) -> A.Compose:
    """
    Instantiates training augmentations using the Albumentations library.

    :param tile_width: Width of tiles input to the model.
    :param tile_height: Height of tiles input to the model.
    :return: Composition of training augmentations.
    """
    return A.Compose([
        A.ColorJitter(p=1, saturation=0.1, hue=0.1,
                      brightness=(0.7, 1.4),
                      contrast=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1),
        A.Transpose(p=0.5),
        A.Perspective(p=1),
        A.OneOf([
            A.GridDistortion(p=1, distort_limit=0.05),
            A.GaussNoise(p=1, var_limit=(10, 100)),
            A.OpticalDistortion(p=1),
            A.OneOf([
                A.PixelDropout(dropout_prob=0.03, drop_value=0, p=1,
                               mask_drop_value=0),
                A.PixelDropout(dropout_prob=0.03, drop_value=255, p=1,
                               mask_drop_value=0)
            ], p=1),
            A.CoarseDropout(p=1, max_width=max(1, int(0.1 * tile_width)),
                            max_height=max(1, int(0.1 * tile_height)),
                            min_width=max(1, int(0.05 * tile_width)),
                            min_height=max(1, int(0.05 * tile_height))),
            A.CoarseDropout(p=1, max_width=max(1, int(0.1 * tile_width)),
                            max_height=max(1, int(0.1 * tile_height)),
                            min_width=max(1, int(0.05 * tile_width)),
                            min_height=max(1, int(0.05 * tile_height)),
                            fill_value=255)
        ], p=1),
        A.OneOf([
            A.Downscale(0.7, 0.95, interpolation=dict(
                downscale=INTER_AREA, upscale=INTER_CUBIC), p=1),
            A.RandomResizedCrop(
                tile_height, tile_width,
                scale=(0.40, 1), ratio=(0.8, 1.2), p=1)], p=0.5)
    ], p=1)


def register_train_augmentations(tile_width: int, tile_height: int,
                                 SCALE: float | None = None,
                                 SCALE_METHOD: int | None = None) -> None:
    """
    Registers training augmentations to MMSegmentation.
    If ``SCALE`` and ``SCALE_METHOD`` are given, a second augmentation
    step is registered to scale input tiles.

    :param tile_width: Width of tiles input to the model.
    :param tile_height: Height of tiles input to the model.
    :param SCALE: Scale factor for tiles (optional).
    :param SCALE_METHOD: Scale method for tiles (optional).
    """
    scale_augmentation, tile_width, tile_height = \
        get_scale_augmentation(tile_width, tile_height,
                               SCALE, SCALE_METHOD)

    @TRANSFORMS.register_module()
    class ScaleAugmentation(BaseTransform):
        """
        Scale augmentation for training.
        """

        def __init__(self):
            super().__init__()

        def transform(self, results: dict) -> dict:
            """
            Applies scaling to the tile and label given in ``results``.

            :param results: Sample from the current training batch.
            :return: Modified sample with scaled tiles.
            """
            results['img_shape'] = (tile_height, tile_width)
            results['ori_shape'] = (tile_height, tile_width)

            image = results["img"]
            mask = results["gt_seg_map"]
            result = scale_augmentation(image=image, mask=mask)
            results["img"] = result["image"]
            results["gt_seg_map"] = result["mask"]
            return results

    augmentations = get_train_augmentations(tile_width, tile_height)

    @TRANSFORMS.register_module()
    class SelectiveAugmentation(BaseTransform):
        """
        Training augmentations such as color shift, cropping, distortion, etc.
        """

        def __init__(self):
            super().__init__()

        def transform(self, results: dict) -> dict:
            """
            Applies augmentations to the tile and label given in ``results``.
            Albumentations intelligently decides which augmentations to also
            apply to the labels, and which to only apply to the image.

            :param results: Sample from the current training batch.
            :return: Modified sample with augmented tiles.
            """
            image = results["img"]
            mask = results["gt_seg_map"]

            result = augmentations(image=image, mask=mask)

            results["img"] = result["image"]
            results["gt_seg_map"] = result["mask"]

            return results


def register_eval_augmentations(tile_width: int, tile_height: int,
                                SCALE: float | None = None,
                                SCALE_METHOD: int | None = None) -> None:
    """
    Registers evaluation (validation, testing) augmentations to MMSegmentation.
    Only relevant if ``SCALE`` and ``SCALE_METHOD`` are given, to register an
    augmentation step which scales input tiles.

    :param tile_width: Width of tiles input to the model.
    :param tile_height: Height of tiles input to the model.
    :param SCALE: Scale factor for tiles (optional).
    :param SCALE_METHOD: Scale method for tiles (optional).
    """
    scale_augmentation, tile_width, tile_height = \
        get_scale_augmentation(tile_width, tile_height,
                               SCALE, SCALE_METHOD)

    @TRANSFORMS.register_module()
    class EvalScaleAugmentation(BaseTransform):
        """
        Scale augmentation for evaluation (validation and testing).
        """

        def __init__(self):
            super().__init__()

        def transform(self, results: dict) -> dict:
            """
            Applies scaling to the tile and label given in ``results``.

            :param results: Sample from the current training batch.
            :return: Modified sample with scaled tiles.
            """
            results['img_shape'] = (tile_height, tile_width)
            results['ori_shape'] = (tile_height, tile_width)
            image = results["img"]
            if "gt_seg_map" in results:
                mask = results["gt_seg_map"]
                result = scale_augmentation(image=image, mask=mask)
                results["img"] = result["image"]
                results["gt_seg_map"] = result["mask"]
            else:
                results["img"] = scale_augmentation(image=image)["image"]

            return results


def get_scale_augmentation(
        tile_width, tile_height,
        SCALE: float | None = None,
        SCALE_METHOD: int | None = None) -> tuple[BasicTransform, int, int]:
    """
    Returns the scale augmentation for the given ``SCALE`` and
    ``SCALE_METHOD``, and the modified ``tile_width`` and ``tile_height``.
    If ``SCALE`` and ``SCALE_METHOD`` are not given, the returned augmentation
    does nothing, and ``tile_width`` and ``tile_height`` are not changed.

    :param tile_width: Width of tiles input to the model.
    :param tile_height: Height of tiles input to the model.
    :param SCALE: Scale factor for tiles (optional).
    :param SCALE_METHOD: Scale method for tiles (optional).
    :return: Tuple of scale augmentation, tile width, and tile height.
    """
    if SCALE:
        if SCALE_METHOD == 1:
            tile_width = int(tile_width * SCALE)
            tile_height = int(tile_height * SCALE)
            scale_augmentation = A.Resize(tile_width, tile_height,
                                          INTER_AREA, p=1)
        else:
            scale_augmentation = A.Downscale(SCALE, SCALE, interpolation=dict(
                downscale=INTER_AREA, upscale=INTER_CUBIC), p=1)
    else:
        scale_augmentation = A.NoOp(p=1)
    return scale_augmentation, tile_width, tile_height


def register_dataset(class_names: tuple[str],
                     palette: list[list[int]]) -> None:
    """
    Registers the segmentation dataset used for training to MMSegmentation.

    :param class_names: Names for each class
    :param palette: List of lists, where each sublist
        contains three integers for R, G, and B color values.
        The sublist at index ``i`` corresponds to the class with name
        ``class_names[i]``.
    """

    @DATASETS.register_module()
    class CustomSegDataset(BaseSegDataset):
        """
        A simple segmentation dataset, using .jpg images and .png masks.
        """
        METAINFO = {"classes": class_names,
                    "palette": palette}

        def __init__(self, **kwargs):
            super().__init__(
                img_suffix='.jpg',
                seg_map_suffix='.png',
                **kwargs)


def register_inference_hook(journal: Journal) -> None:
    """
    Registers a callback for model inference, to allow capture of model logits
    and to register which predictions have already been computed.

    :param journal: Journal to save inference progress.
    """

    @HOOKS.register_module()
    class InferenceHook(Hook):
        """
        Callback for model inference.
        """

        def __init__(self, prediction_logits_dir: str | None = None):
            self.prediction_logits_dir: str | None = prediction_logits_dir

        def _after_iter(self, runner, batch_idx: int,
                        data_batch: DATA_BATCH | None = None,
                        outputs: Sequence | dict | None = None,
                        mode: str = 'train') -> None:
            """
            Callback invoked after each inference step.

            :param runner: MMSegmentation test runner.
            :param batch_idx: Index of the current batch.
            :param data_batch: Auxiliary information for the given batch.
            :param outputs: Batch containing predictions and logits.
            :param mode: Operation mode (``'train'``, ``'val'``, or
                ``'test'``). Callback is only executed if ``mode`` is
                ``'test'``.
            """
            if mode == 'train' or mode == 'val':
                return

            # logits = []
            for x in outputs:
                filename = os.path.splitext(os.path.basename(x.img_path))[0]

                # Example how to extract logits from model
                # logit = x.seg_logits.data
                # logits.append(logit)
                # logit_path = os.path.join(
                #     self.prediction_logits_dir, filename)
                #
                # # For each pixel, get the highest logit and save it
                # # in compressed format
                # logit = np.max(logit.detach().cpu().numpy(), axis=0)
                #
                # np.savez_compressed(logit_path, logit=logit)

                # Register progress for mode inference
                journal[filename] = None


def deregister_all() -> None:
    """
    De-registers all custom components from MMSegmentation.
    This should be invoked when finishing an operation, because registering
    a component twice will raise an exception.
    """
    if 'CustomSegDataset' in DATASETS.module_dict:
        del DATASETS.module_dict['CustomSegDataset']

    if 'SelectiveAugmentation' in TRANSFORMS.module_dict:
        del TRANSFORMS.module_dict['SelectiveAugmentation']

    if 'ScaleAugmentation' in TRANSFORMS.module_dict:
        del TRANSFORMS.module_dict['ScaleAugmentation']

    if 'EvalScaleAugmentation' in TRANSFORMS.module_dict:
        del TRANSFORMS.module_dict['EvalScaleAugmentation']

    if 'WeightedInfiniteSampler' in DATA_SAMPLERS.module_dict:
        del DATA_SAMPLERS.module_dict['WeightedInfiniteSampler']

    if 'InferenceHook' in HOOKS.module_dict:
        del HOOKS.module_dict['InferenceHook']


def register_train_components(train_task, train_dataset_task) -> None:
    """
    Registers custom MMSegmentation components needed for training.

    :param train_task: TrainTask to run.
    :param train_dataset_task: DatasetTask used for training.
    """
    register_dataset(tuple(train_dataset_task.label_legend.keys()),
                     train_dataset_task.palette)

    # Color shift, geometric transformations, etc.
    register_train_augmentations(train_dataset_task.tile_width,
                                 train_dataset_task.tile_height,
                                 train_task.SCALE,
                                 train_task.SCALE_METHOD)

    # For validation
    register_eval_augmentations(train_dataset_task.tile_width,
                                train_dataset_task.tile_height,
                                train_task.SCALE,
                                train_task.SCALE_METHOD)

    # Weighted random sampler to mitigate class imbalance
    register_sampler(joblib_load_typed(train_task.sample_weights_file, list))


def register_eval_components(train_task,
                             train_dataset_task) -> None:
    """
    Registers custom MMSegmentation components needed for testing.

    :param train_task: TrainTask which trained the model to evaluate.
    :param train_dataset_task: DatasetTask used for training.
    """
    register_dataset(tuple(train_dataset_task.label_legend.keys()),
                     train_dataset_task.palette)
    register_eval_augmentations(train_dataset_task.tile_width,
                                train_dataset_task.tile_height,
                                train_task.SCALE,
                                train_task.SCALE_METHOD)


def register_predict_components(train_task, train_dataset_task,
                                predict_journal) -> None:
    """
    Registers custom MMSegmentation components needed for training.

    :param train_task: TrainTask which trained the model to use for inference.
    :param train_dataset_task: DatasetTask used for training.
    :param predict_journal: Journal to register inference progress.
    """
    register_dataset(tuple(train_dataset_task.label_legend.keys()),
                     train_dataset_task.palette)
    register_eval_augmentations(train_dataset_task.tile_width,
                                train_dataset_task.tile_height,
                                train_task.SCALE,
                                train_task.SCALE_METHOD)
    register_inference_hook(predict_journal)
