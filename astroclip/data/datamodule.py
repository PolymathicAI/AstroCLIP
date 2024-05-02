from typing import Callable, Dict, List

import datasets
import lightning as L
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import CenterCrop

from astroclip.astrodino.data.dataset import dr2_rgb


class AstroClipDataloader(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        columns: List[str] = ["image", "spectrum"],
        batch_size: int = 512,
        num_workers: int = 10,
        collate_fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        self.dataset = datasets.load_from_disk(self.hparams.path)
        self.dataset.set_format(type="torch", columns=self.hparams.columns)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,  # NOTE: disable for debugging
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )


class AstroClipCollator:
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

    def __call__(self, samples):
        # collate and handle dimensions
        samples = default_collate(samples)
        # process images
        samples["image"] = self._process_images(samples["image"])
        return samples
