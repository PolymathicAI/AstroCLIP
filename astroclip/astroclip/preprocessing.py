import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import CenterCrop

from astroclip.astrodino.data.datasets.legacysurvey import dr2_rgb


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
