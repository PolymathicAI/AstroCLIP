from typing import Callable, Dict, List

import lightning as L
import torch
from torch import Tensor

import datasets


class AstroClipDataset(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        columns: List[str],
        batch_size: int,
        num_workers: int,
        collate_fn: Callable[Dict[str, Tensor], Dict[str, Tensor]] = None,
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
