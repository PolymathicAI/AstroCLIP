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
        x: Dict[str, Dict[str, torch.FloatTensor]],
        image: bool = True,
        return_weights: bool = False,
    ):
        # Embed image
        if image:
            embedding = self.image_encoder((x, None))

        # Embed spectrum
        else:
            embedding = self.spectrum_encoder(x, return_weights=return_weights)

        return embedding

    def training_step(self, batch, batch_idx):
        im, sp, _ = batch
        image_features = self.image_encoder((im.cuda(), None))
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
        im, sp, _ = batch
        image_features = self.image_encoder((im.cuda(), None))
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
