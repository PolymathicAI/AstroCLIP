from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule as L
from torch.optim import lr_scheduler

from astroclip.astroclip.loss import CLIPLoss

__all__ = ["AstroCLIP"]


class AstroCLIP(L.LightningModule):
    def __init__(
        self,
        image_encoder: nn.Module,
        spectrum_encoder: nn.Module,
        temperature: float,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 100,
        eta_min: float = 5e-7,
        logit_scale: float = 15.5,
        learnable_logit_scale: bool = False,
    ):
        """
        The AstroCLIP model that takes an image and a spectrum and embeds them into a common space using CLIP loss.
        Note that you must provide the image and spectrum encoders to be used for the embedding.

        Args:
            image_encoder (nn.Module): The image encoder to be used for embedding.
            spectrum_encoder (nn.Module): The spectrum encoder to be used for embedding.
            temperature (float): The temperature parameter for the CLIP loss.
            lr (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            epochs (int): The number of epochs for training.
            eta_min (float): The minimum learning rate for the scheduler.
            logit_scale (float): The logit scale for the CLIP loss.
            learnable_logit_scale (bool): Whether the logit scale should be learnable.
        """
        super().__init__()
        self.save_hyperparameters()

        # Define the image and spectrum encoder
        self.image_encoder = image_encoder
        self.spectrum_encoder = spectrum_encoder

        # Logit scale is fixed to 15.5 and is not a learnable parameter
        if not learnable_logit_scale:
            self.logit_scale = np.log(logit_scale)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))

        # Use CLIP loss
        self.criterion = CLIPLoss()

    def forward(
        self,
        batch: Dict[str, Dict[str, torch.FloatTensor]],
        return_weights: bool = False,
    ):
        im, sp = batch["image"], batch["spectrum"]

        if not return_weights:
            image_features = self.image_encoder(im)
            spectrum_features = self.spectrum_encoder(sp)
            return {"image": image_features, "spectrum": spectrum_features}

        else:
            image_features, image_weights = self.image_encoder(im, return_weights=True)
            spectrum_features, spectrum_weights = self.spectrum_encoder(
                sp, return_weights=True
            )
            return {
                "image": image_features,
                "spectrum": spectrum_features,
                "image_weights": image_weights,
                "spectrum_weights": spectrum_weights,
            }

    def training_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]
        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)
        loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )
        self.log("train_loss_withlogit", loss_withlogit)
        loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )
        self.log("train_loss_nologit", loss_nologit)
        self.log("scale", self.logit_scale)
        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]
        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)
        val_loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )
        self.log("val_loss_nologit", val_loss_nologit)
        val_loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )
        self.log("val_loss_withlogit", val_loss_withlogit)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'step' for step-wise updating
                "frequency": 1,  # how often to apply
            },
        }
