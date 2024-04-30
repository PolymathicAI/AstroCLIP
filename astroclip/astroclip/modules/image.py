import torch
import torch.nn as nn

from astroclip.astroclip.modules.common import MLP, CrossAttentionHead
from astroclip.astrodino.dinov2.dinov2.eval.setup import setup_and_build_model


def forward_image_backbone(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the image transformer model to get all tokens."""
    x = self.patch_embed(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x


class ImageModule(nn.Module):
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

        class config:
            output_dir = save_directory
            config_file = config
            pretrained_weights = model_weights
            opts = []

        # Define DINO model
        self.backbone, _ = setup_and_build_model(config())
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
