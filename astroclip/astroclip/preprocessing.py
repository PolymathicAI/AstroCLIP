import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import CenterCrop

from astroclip.astrodino.data.datasets.legacysurvey import dr2_rgb
from astroclip.specformer.preprocessing import preprocess as preprocess_spectrum


class ImageSpectrumCollator:
    def __init__(self, center_crop: int = 144):
        self.center_crop = CenterCrop(center_crop)

    def _process_images(self, images):
        # convert to rgb
        img_outs = []
        for img in images:
            rgb_img = torch.tensor(
                dr2_rgb(img.permute(2, 0, 1), bands=["g", "r", "z"])[None, :, :, :]
            )
            img_outs.append(rgb_img)
        images = torch.concatenate(img_outs)

        images = self.center_crop(images.permute(0, 3, 2, 1))
        return images

    def _process_spectra(self, spectra):
        # slice the spectra
        spectra = [
            preprocess_spectrum(spectrum)
            for spectrum in spectra
            if np.array(spectrum).std() > 0
        ]

        # pad the spectra
        spectra = pad_sequence(
            [torch.tensor(spectrum) for spectrum in spectra],
            batch_first=True,
            padding_value=0,
        )

        return spectra

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)

        # process images
        samples["image"] = self._process_images(samples["image"])

        # process spectra
        samples["spectrum"] = self._process_spectra(samples["spectrum"])

        return samples
