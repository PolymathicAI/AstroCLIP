# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms
from typing import Sequence
import numpy as np
import torch

from astroclip.astrodino.data.astro_augmentations import RandomGaussianBlur, RandomGaussianNoise, ToRGB

logger = logging.getLogger("dinov2")

class DataAugmentationAstroDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomCrop(
                    global_crops_size
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomCrop(
                    local_crops_size
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        global_transfo1_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=1.0),
                RandomGaussianNoise(p=1.0, im_dim=global_crops_size)
            ]
        )
        

        global_transfo2_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.1),
                RandomGaussianNoise(p=0.1, im_dim=global_crops_size),
            ]
        )

        local_transfo_extra = transforms.Compose(
            [
                RandomGaussianBlur(p=0.5),
                RandomGaussianNoise(p=0.5, im_dim=local_crops_size),
            ]
        )

        to_rgb = ToRGB()
        
        self.global_transfo1 = transforms.Compose([global_transfo1_extra, to_rgb])
        self.global_transfo2 = transforms.Compose([global_transfo2_extra, to_rgb])
        self.local_transfo = transforms.Compose([local_transfo_extra, to_rgb])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = np.array(self.geometric_augmentation_global(image))
        global_crop_1 = torch.tensor(self.global_transfo1(im1_base)).permute(2,0,1)

        im2_base = np.array(self.geometric_augmentation_global(image))
        global_crop_2 = torch.tensor(self.global_transfo2(im2_base)).permute(2,0,1)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            torch.tensor(self.local_transfo(np.array(self.geometric_augmentation_local(image)))).permute(2,0,1) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output