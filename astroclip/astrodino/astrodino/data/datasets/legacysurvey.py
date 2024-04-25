# Dataset file for DESI Legacy Survey data
import logging
import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union
from jaxtyping import Float, Str, Array, List

import h5py
import numpy as np
from PIL import Image as im
from torchvision.datasets import VisionDataset
import numpy as np
from numpy import ndarray

"""
Transformation from raw image data (nanomaggies) to the rgb values displayed
at the legacy viewer https://www.legacysurvey.org/viewer

Code copied from
https://github.com/legacysurvey/imagine/blob/master/map/views.py
"""

def sdss_rgb(imgs: Float[Array, "b h w c"], bands: List[Str], scales=None, m=0.02):
    """Convert images to RGB using SDSS imaging color scheme."""
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


def dr2_rgb(rimgs: Float[Array, "b h w c"], bands: List[Str], **ignored):
    """Convert images to RGB using DECaLS imaging color scheme."""
    return sdss_rgb(
        rimgs, bands, scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03
    )


logger = logging.getLogger("astrodino")
_Target = float


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        # Using the North Galactic Cap dataset
        split_lengths = {
            _Split.TRAIN: 9_500_000, 
            _Split.VAL: 100_000,       
            _Split.TEST: 400_000,
        }
        return split_lengths[self]


class LegacySurvey(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

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
        """
        Dataset for the DECaLS Legacy Survey data from Stein, et al. (2020).

        Args:
            split: The split of the dataset to use.
            root: The root directory where the dataset is stored.
        """
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        # We start by opening the hdf5 files located at the root directory
        self._files = [
            h5py.File(
                os.path.join(
                    root, "images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1)
                )
            )
            for i in range(10)
        ]

        # Create randomized array of indices
        rng = np.random.default_rng(seed=42)
        self._indices = rng.permutation(int(1e7))
        if split == LegacySurvey.Split.TRAIN.value:
            self._indices = self._indices[:9_500_000]
        elif split == LegacySurvey.Split.VAL.value:
            self._indices = self._indices[9_500_000:-400_000]
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

        # For testing, let's convert image to PIL images
        image = im.fromarray(
            (dr2_rgb(image, bands=["g", "r", "z"]) * 255).astype("uint8")
        )

        # Apply transformations
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._indices)
