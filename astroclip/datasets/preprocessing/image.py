import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import CenterCrop


class ImageCollator:
    def __init__(self, center_crop: int = 144):
        self.center_crop = CenterCrop(center_crop)

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)
        images = samples["image"]  # h x w x c

        # convert to rgb
        img_outs = []
        for img in images:
            rgb_img = torch.tensor(dr2_rgb(img.T, bands=["g", "r", "z"])[None, :, :, :])
            img_outs.append(rgb_img)
        images = torch.concatenate(img_outs)

        # center crop
        images = self.center_crop(images.permute(0, 3, 2, 1))

        # return
        samples["image"] = images
        return samples


def sdss_rgb(imgs, bands, scales: dict = None, m: int = 0.02):
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


def dr2_rgb(rimgs, bands: list, **ignored):
    """Convert a set of DECaLS Legacy Survey images to an RGB image."""
    return sdss_rgb(
        rimgs, bands, scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03
    )
