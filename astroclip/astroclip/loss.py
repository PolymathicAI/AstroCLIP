import torch
from torch import nn
import torch.nn.functional as F

from jaxtyping import Array
from typing import Tuple

class CLIPLoss(nn.Module):
    def get_logits(
            self, image_features: Array[torch.FloatTensor, "b c"], spectrum_features: Array[torch.FloatTensor, "b c"], logit_scale: float
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        return logits_per_image, logits_per_image.T

    def forward(
        self, image_features: Array[torch.FloatTensor, "b c"], spectrum_features: Array[torch.FloatTensor, "b c"], logit_scale: float, output_dict: bool = False
    ) -> torch.FloatTensor:
        """Calculate the contrastive loss for the given image and spectrum features."""
        
        logits_per_image, logits_per_spectrum = self.get_logits(
            image_features, spectrum_features, logit_scale
        )
        labels = torch.arange(
            logits_per_image.shape[0], device=image_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_spectrum, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss
