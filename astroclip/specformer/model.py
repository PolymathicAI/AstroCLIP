import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer


class SpecFormer(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embed = nn.Linear(self.hparams.input_dim, self.hparams.embed_dim)

        self.head = nn.Linear(self.hparams.embed_dim, 1)

        self.abs_pos = nn.Embedding(self.hparams.max_len, self.hparams.embed_dim)

        trans_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.embed_dim,
            nhead=self.hparams.num_heads,
            dim_feedforward=4 * self.hparams.embed_dim,
            dropout=self.hparams.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=self.hparams.norm_first,
        )

        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=trans_layer,
            num_layers=self.hparams.num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, input):
        t = input.shape[1]

        if len(input.shape) == 2:
            input = input.unsqueeze(-1)

        x = self.embed(input) + self.abs_pos.weight[:t].unsqueeze(0)

        x = F.gelu(self.encoder_stack(x))

        # adding the input back in so we model the difference
        x = input + self.head(x)

        return x

    def training_step(self, batch):
        input = batch["input"]
        target = batch["target"]

        output = self(input)
        loss = F.mse_loss(output, target)
        self.log("training_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        input = batch["input"]
        target = batch["target"]

        output = self(input)
        loss = F.mse_loss(output, target)
        self.log("val_training_loss", loss, prog_bar=True)
        return loss
