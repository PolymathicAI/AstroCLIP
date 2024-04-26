import torch
import torch.nn as nn

from astroclip.astroclip.utils import (
    fnc,
    forward_image_backbone,
    forward_spectrum_backbone,
    load_spectrum_model_from_ckpt,
)
from astroclip.astrodino.dinov2.dinov2.eval.setup import setup_and_build_model


class ImageModule(nn.Module):
    def __init__(
        self,
        save_directory: str,
        config: str,
        model_weights: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 1024,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        class config:
            output_dir = save_directory
            config_file = config
            pretrained_weights = model_weights
            opts = []

        # Define DINO model
        self.backbone, dtype = setup_and_build_model(config())
        self.backbone.forward = forward_image_backbone.__get__(self.backbone)

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
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.tensor, return_weights: bool = False):
        # Pass through the backbone
        if self.freeze_backbone:
            with torch.no_grad():
                embedding = self.backbone(x)
        else:
            embedding = self.backbone(x)

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x += self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()


class SpectrumModule(nn.Module):
    def __init__(
        self,
        save_path: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Load the model from the checkpoint
        out = self._load_model_from_ckpt(save_path)
        self.config = out["config"]
        self.backbone = out["model"]

        # Replace forward pass to give all tokens
        self.backbone.forward = forward_spectrum_backbone.__get__(self.backbone)

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
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        # Slice the spectrum
        x = fnc(x.unsqueeze(-1))

        # Embed the spectrum using the pretrained model
        if self.freeze_backbone:
            with torch.no_grad():
                embedding = self.backbone(x)["embedding"]
        else:
            embedding = self.backbone(x)["embedding"]

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x += self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()


class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, kwargs["embed_dim"]))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=kwargs["embed_dim"],
            num_heads=kwargs["n_head"],
            batch_first=True,
            kdim=kwargs["model_embed_dim"],
            vdim=kwargs["model_embed_dim"],
        )
        self.layernorm = nn.LayerNorm(kwargs["embed_dim"])
        self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self, x: torch.tensor):
        batch_size = x.shape[0]
        attentions = self.multihead_attn(
            query=self.query.repeat(batch_size, 1, 1),
            key=x,
            value=x,
            need_weights=False,
            average_attn_weights=False,
        )[0]
        x = self.layernorm(self.dropout(attentions))
        return x, attentions[1]


class MLP(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.d_model = kwargs["embed_dim"]
        self.dim_feedforward = self.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(kwargs["dropout"]),
            nn.Linear(self.dim_feedforward, self.d_model),
        )

    def forward(self, x: torch.tensor):
        return self.mlp(x)
