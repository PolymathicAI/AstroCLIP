import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astroclip.modules.common import MLP, CrossAttentionHead


def load_model_from_ckpt(ckpt_path: str):
    # TODO: Get rid of, deprecated from fi-llm
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

    return {"model": model_, "config": config}


def fnc(x):
    # TODO: Get rid of, deprecated from fi-llm
    std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
    x = (x - mean) / std
    x = slice(x, 20, 10)
    x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
    x[:, 0, 0] = (mean.squeeze() - 2) / 2
    x[:, 0, 1] = (std.squeeze() - 2) / 8
    return x


def slice(x, section_length=10, overlap=5):
    # TODO: Get rid of, deprecated from fi-llm
    start_indices = np.arange(0, x.shape[1] - overlap, section_length - overlap)
    sections = [
        x[:, start : start + section_length].transpose(1, 2) for start in start_indices
    ]
    # If the last section is not of length 'section_length', you can decide whether to keep or discard it
    if sections[-1].shape[1] < section_length:
        sections.pop(-1)  # Discard the last section
    return torch.cat(sections, 1)


def forward_specformer(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the spectrum transformer model to get all tokens."""
    pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)  # shape (t)

    # forward the GPT model itself
    data_emb = self.data_embed(x)  # to shape (b, t, embedding_dim)
    pos_emb = self.position_embed(pos)  # to shape (t, embedding_dim)

    x = self.dropout(data_emb + pos_emb)
    for block in self.blocks:
        x = block(x)
    x = self.final_layernorm(x)
    return x.detach().clone()


class SpectrumHead(nn.Module):
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
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
        out = load_model_from_ckpt(model_path)
        self.backbone = out["model"]

        # Set up the backbone forward pass
        self.backbone.forward = forward_specformer.__get__(self.backbone)

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
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        x = fnc(x)

        # Embed the spectrum using the pretrained model
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
