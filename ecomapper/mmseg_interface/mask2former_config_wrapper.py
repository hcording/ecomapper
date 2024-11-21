"""
Wrapper for the MMSegmentation implementation of Mask2Former.
"""

import os

from ecomapper.mmseg_interface.mmseg_config_wrapper import MMSegConfigWrapper


class Mask2FormerConfigWrapper(MMSegConfigWrapper):

    @staticmethod
    def get_metafile() -> str:
        return os.path.join("mask2former", "metafile.yaml")

    @staticmethod
    def get_model_name():
        return "Mask2Former"

    def __init__(self, variant_config_file: str, tile_width: int,
                 tile_height: int, num_classes: int, train_task):
        super().__init__(variant_config_file,
                         tile_width, tile_height, num_classes,
                         train_task)

        self.mmseg_config.crop_size = (self.tile_width, self.tile_height)
        self.mmseg_config.model.data_preprocessor.size = (
            self.mmseg_config.crop_size)
        self.mmseg_config.data_preprocessor.size = self.mmseg_config.crop_size
        self.mmseg_config.model.data_preprocessor.size = (
            self.mmseg_config.crop_size)

        self.mmseg_config.model.decode_head.num_classes = num_classes
        self.mmseg_config.num_classes = num_classes

        # Based on paper, "no object tokens" get weight of 0.1
        self.mmseg_config.model.decode_head.loss_cls["class_weight"] = \
            [1.0] * self.num_classes + [0.1]
