import numpy as np
import torch
from jaxtyping import Array, Dict, List
from torch.utils.data import default_collate
from torchvision.transforms import CenterCrop


class ImageCollator:
    def __init__(
        self,
        center_crop: int = 144,
        bands: List = ["g", "r", "z"],
        scales: Dict = dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)),
        m: float = 0.03,
    ):
        self.bands = bands
        self.scales = scales
        self.m = m
        self.center_crop = CenterCrop(center_crop)

    def __call__(self, samples):
        images = samples["images"]
        images = self.center_crop(images)

        # Center crop the images
        if len(images.shape) == 3:
            return dr2_rgb(images.T, self.bands).T
        if len(images.shape) == 4:
            img_outs = []
            for img in images:
                img_outs.append(dr2_rgb(img.T, self.bands).T[None, :, :, :])
            return torch.concatenate(img_outs)

        samples["images"] = images
        return samples


def sdss_rgb(
    imgs: Array[np.ndarray, "h w c"],
    bands: List["c"],
    scales: Dict = None,
    m: int = 0.02,
):
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


def dr2_rgb(rimgs: Array[np.ndarray, "h w c"], bands: List["c"], **ignored):
    """Convert a set of Legacy Survey images to an RGB image."""
    return sdss_rgb(
        rimgs, bands, scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03
    )
