import torch
from torch import nn


class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        """Multihead cross-attention layer with dropout."""
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
        """MLP with GELU activation and dropout."""
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
