import os
import sys
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov2.eval.setup import setup_and_build_model

from ..modules import MLP, CrossAttentionHead
from .specformer import SpecFormer


class AstroClipModel(L.LightningModule):
    def __init__(
        self,
        image_encoder: nn.Module,
        spectrum_encoder: nn.Module,
        temperature: float = 15.5,
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
        input: torch.Tensor,
        input_type: str,
    ):
        if input_type == "image":
            return self.image_encoder(input)

        elif input_type == "spectrum":
            return self.spectrum_encoder(input)

        else:
            raise ValueError("Input type must be either 'image' or 'spectrum'")

    def training_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]

        # Get the image and spectrum features
        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)

        # Calculate the CLIP loss
        loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )
        loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )

        # Log the losses
        self.log("train_loss_withlogit", loss_withlogit)
        self.log("train_loss_nologit", loss_nologit)
        self.log("scale", self.logit_scale)

        # Return the loss
        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        im, sp = batch["image"], batch["spectrum"]

        # Get the image and spectrum features
        image_features = self.image_encoder(im)
        spectrum_features = self.spectrum_encoder(sp)

        # Calculate the CLIP loss
        val_loss_nologit = self.criterion(
            image_features, spectrum_features, self.hparams.logit_scale
        )
        val_loss_withlogit = self.criterion(
            image_features, spectrum_features, self.hparams.temperature
        )

        # Log the losses
        self.log("val_loss_nologit", val_loss_nologit)
        self.log("val_loss_withlogit", val_loss_withlogit)


class CLIPLoss(nn.Module):
    def get_logits(
        self,
        image_features: torch.FloatTensor,
        spectrum_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        return logits_per_image, logits_per_image.T

    def forward(
        self,
        image_features: torch.FloatTensor,
        spectrum_features: torch.FloatTensor,
        logit_scale: float,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the image and spectrum features
        logits_per_image, logits_per_spectrum = self.get_logits(
            image_features, spectrum_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_image.shape[0], device=image_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_spectrum, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss


class ImageHead(nn.Module):
    def __init__(
        self,
        config: str,
        model_weights: str,
        save_directory: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        """
        Cross-attention image module that takes token outputs from the AstroDINO model and passes them through a
        cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_directory (str): Path to the directory containing the AstroDINO model.
            config (str): Path to the configuration file of the AstroDINO model.
            model_weights (str): Path to the weights of the AstroDINO model.
            embed_dim (int): Dimension of the AstroCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the AstroDINO embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the AstroDINO model.
        """
        super().__init__()

        # Define DINO config
        class config:
            output_dir = save_directory
            config_file = config
            pretrained_weights = model_weights
            opts = []

        # Define DINO model
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to null
        self.backbone, _ = setup_and_build_model(config())
        sys.stdout = sys.__stdout__  # Reset stdout

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # Set up MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.tensor, return_weights: bool = False):
        # Pass through the backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            x = self.backbone.patch_embed(x)
            for blk in self.backbone.blocks:
                x = blk(x)
            embedding = self.backbone.norm(x)

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x = self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()


class SpectrumHead(nn.Module):
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        """
        Cross-attention spectrum module that takes a spectrum and passes it through a pretrained SpecFormer model and
        then through a cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the AstroCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the SpecFormer embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the SpecFormer model.
        """
        super().__init__()
        # Load the model from the checkpoint
        checkpoint = torch.load(model_path)
        self.backbone = SpecFormer(**checkpoint["hyper_parameters"])
        if load_pretrained_weights:
            self.backbone.load_state_dict(checkpoint["state_dict"])

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # Set up MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        # Embed the spectrum using the pretrained model
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["embedding"]

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x = x + self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()
