# Dataset file for DESI Legacy Survey data
import logging
import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image as im
from torchvision.datasets import VisionDataset

logger = logging.getLogger("astrodino")
_Target = float


class _SplitFull(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitFull.TRAIN: 74_500_000,
            _SplitFull.VAL: 100_000,
            _SplitFull.TEST: 400_000,
        }
        return split_lengths[self]


class LegacySurvey(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitFull]

    def __init__(
        self,
        *,
        split: "LegacySurvey.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        # We start by opening the hdf5 files located at the root directory
        self._files = [
            h5py.File(
                os.path.join(
                    root, "north/images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(14)
        ]
        self._files += [
            h5py.File(
                os.path.join(
                    root, "south/images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(61)
        ]

        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        self._indices = rng.permutation(int(7.5e7))
        if split == LegacySurvey.Split.TRAIN.value:
            self._indices = self._indices[:74_500_000]
        elif split == LegacySurvey.Split.VAL.value:
            self._indices = self._indices[74_500_000:-400_000]
        else:
            self._indices = self._indices[-400_000:]

    @property
    def split(self) -> "LegacySurvey.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        true_index = self._indices[index]
        image = self._files[true_index // int(1e6)]["images"][
            true_index % int(1e6)
        ].astype("float32")
        target = None
        image = torch.tensor(dr2_rgb(image, bands=["g", "r", "z"])).permute(2, 0, 1)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._indices)


class _SplitNorth(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitNorth.TRAIN: 13_500_000,
            _SplitNorth.VAL: 100_000,
            _SplitNorth.TEST: 400_000,
        }
        return split_lengths[self]


class LegacySurveyNorth(VisionDataset):
    Target = Union[_Target]
    Split = Union[_SplitNorth]

    def __init__(
        self,
        *,
        split: "LegacySurvey.Split",
        root: str,
        extra: str = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        # We start by opening the hdf5 files located at the root directory
        self._files = [
            h5py.File(
                os.path.join(
                    root, "north/images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(14)
        ]

        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        self._indices = rng.permutation(int(1.4e7))
        if split == LegacySurvey.Split.TRAIN.value:
            self._indices = self._indices[:13_500_000]
        elif split == LegacySurvey.Split.VAL.value:
            self._indices = self._indices[13_500_000:-400_000]
        else:
            self._indices = self._indices[-400_000:]

    @property
    def split(self) -> "LegacySurvey.Split":
        return self._split

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        true_index = self._indices[index]
        image = self._files[true_index // int(1e6)]["images"][
            true_index % int(1e6)
        ].astype("float32")
        target = None
        image = torch.tensor(dr2_rgb(image, bands=["g", "r", "z"])).permute(2, 0, 1)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._indices)


def sdss_rgb(imgs, bands, scales=None, m=0.02):
    """
    Transformation from raw image data (nanomaggies) to the rgb values displayed
    at the legacy viewer https://www.legacysurvey.org/viewer

    Code copied from
    https://github.com/legacysurvey/imagine/blob/master/map/views.py
    """
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
