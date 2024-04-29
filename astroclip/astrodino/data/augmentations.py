# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Sequence

from dinov2.data.transforms import GaussianBlur, make_normalize_transform
from torchvision import transforms

logger = logging.getLogger("dinov2")

# No normalization by default
DESI_DEFAULT_MEAN = (0.0, 0.0, 0.0)
DESI_DEFAULT_STD = (1.0, 1.0, 1.0)


"""
Transformation from raw image data (nanomaggies) to the rgb values displayed
at the legacy viewer https://www.legacysurvey.org/viewer

Code copied from
https://github.com/legacysurvey/imagine/blob/master/map/views.py
"""


def sdss_rgb(imgs, bands, scales=None, m=0.02):
    rgbscales = {
        "u": (2, 1.5),  # 1.0,
        "g": (2, 2.5),
        "r": (1, 1.5),
        "i": (0, 1.0),
        "z": (0, 0.4),  # 0.3
    }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.0) * 1e-6
    H, W = I.shape
    rgb = np.zeros((H, W, 3), np.float32)
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        rgb[:, :, plane] = (img * scale + m) * fI / I
    rgb = np.clip(rgb, 0, 1)
    return rgb


def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(
        rimgs, bands, scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03
    )


def make_normalize_transform(
    mean: Sequence[float] = DESI_DEFAULT_MEAN,
    std: Sequence[float] = DESI_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


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
                transforms.RandomCrop(global_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomCrop(local_crops_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.33)

        # normalization
        self.normalize = transforms.Compose(
            [
                make_normalize_transform(),
            ]
        )

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
