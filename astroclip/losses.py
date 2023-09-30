import torch, torch.nn as nn, torch.nn.functional as F


class CLIPLoss(nn.Module):
    """ Simple contrastive loss for CLIP
    """
    def get_logits(self, image_features, spectrum_features, logit_scale):
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        logits_per_spectrum = logit_scale * spectrum_features @ image_features.T
        return logits_per_image, logits_per_spectrum

    def forward(self, image_features, spectrum_features, logit_scale, output_dict=False):
        logits_per_image, logits_per_spectrum = self.get_logits(image_features, spectrum_features, logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_spectrum, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss