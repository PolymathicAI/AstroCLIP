import numpy as np
import torch
import torch.nn.functional as F


def forward_image_backbone(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the image transformer model to get all tokens."""
    x = self.patch_embed(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x
