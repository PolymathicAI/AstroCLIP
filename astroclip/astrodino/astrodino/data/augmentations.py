# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Sequence

import torch
from dinov2.data.transforms import GaussianBlur, make_normalize_transform
from torchvision import transforms

logger = logging.getLogger("dinov2")

# ImageNet normalization
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# Normalization Transforms
def make_normalize_transform(
    mean: Sequence[float] = MEAN,
    std: Sequence[float] = STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# Radial position encoding
def generate_radial_position_encoding_torch(height, width, scale=0.1):
    # Generate a grid of coordinates corresponding to the image pixels
    x = torch.linspace(-scale, scale, steps=width)
    y = torch.linspace(-scale, scale, steps=height)
    xv, yv = torch.meshgrid(x, y, indexing="ij")

    # Compute the distance of each pixel from the center
    radial_dist = torch.sqrt(xv**2 + yv**2)

    return radial_dist


class DataAugmentationAstroDINO(object):
    def __init__(
        self,
        global_crops_scale: float,
        local_crops_scale: float,
        local_crops_number: float,
        global_crops_size: float = 224,
        local_crops_size: float = 96,
    ):
        """
        The data augmentation pipeline for AstroDINO.

        Args:
            global_crops_scale: The scale of the global crops.
            local_crops_scale: The scale of the local crops.
            local_crops_number: The number of local crops.
            global_crops_size: The size of the global crops.
            local_crops_size: The size of the local crops.
        """
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
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        self.normalize = transforms.Compose(
            [transforms.ToTensor(), make_normalize_transform()]
        )

        # We just remove color jittering here
        self.global_transfo1 = transforms.Compose(
            [global_transfo1_extra, self.normalize]
        )
        self.global_transfo2 = transforms.Compose(
            [global_transfo2_extra, self.normalize]
        )
        self.local_transfo = transforms.Compose([local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
