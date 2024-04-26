from typing import Optional, Tuple

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


def forward_spectrum_backbone(
    self, x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of the spectrum transformer model to get all tokens."""
    device = x.device
    t = x.shape[1]

    # find the mask locations
    locs = x != y

    if t > self.config.block_size:
        raise ValueError(
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.block_size}"
        )
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

    # forward the GPT model itself
    data_emb = self.data_embed(x)  # to shape (b, t, embedding_dim)
    pos_emb = self.position_embed(pos)  # to shape (t, embedding_dim)

    x = self.dropout(data_emb + pos_emb)
    for block in self.blocks:
        x = block(x)
    x = self.final_layernorm(x)
    embedding = x.detach().clone()

    preds = self.head(x)
    if y is not None:
        # if we are given some desired targets also calculate the loss
        locs = locs.type_as(preds)
        loss = F.mse_loss(preds * locs, y * locs, reduction="mean") / locs.mean()
    else:
        loss = None

    return {"preds": preds, "loss": loss, "embedding": embedding}


def slice(x, section_length: int = 10, overlap: int = 5):
    """Slice the spectrum into sections of length 'section_length' with overlap 'overlap'."""
    start_indices = np.arange(0, x.shape[1] - overlap, section_length - overlap)
    sections = [
        x[:, start : start + section_length].transpose(1, 2) for start in start_indices
    ]

    # If the last section is not of length 'section_length', you can decide whether to keep or discard it
    if sections[-1].shape[1] < section_length:
        sections.pop(-1)  # Discard the last section

    return torch.cat(sections, 1)


def fnc(x):
    """Normalize the spectrum and pad it with zeros."""
    std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
    x = (x - mean) / std
    x = slice(x, 20, 10)
    x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
    x[:, 0, 0] = (mean.squeeze() - 2) / 2
    x[:, 0, 1] = (std.squeeze() - 2) / 8

    return x


def load_spectrum_model_from_ckpt(self, ckpt_path: str):
    """Load a spectrum model from a checkpoint."""
    if Path(ckpt_path).is_dir():
        ckpt_path = Path(ckpt_path) / "ckpt.pt"

    chkpt = torch.load(ckpt_path)
    config = chkpt["config"]
    state_dict = chkpt["model"]
    model_name = config["model"]["kind"]
    model_keys = get_model_keys(model_name)

    model_args = {k: config["model"][k] for k in model_keys}

    model_ctr, config_cls = model_registry[model_name]
    model_config = config_cls(**model_args)
    model_ = model_ctr(model_config)
    model_.load_state_dict(state_dict)
