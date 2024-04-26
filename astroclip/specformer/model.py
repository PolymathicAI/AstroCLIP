import math
import numbers
from typing import Callable, Optional, Tuple, Union

import lightning as L
import torch
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

        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    causal=False,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters_datapt()

    def _reset_parameters_datapt(self):
        # not scaling the initial embeddngs.
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.hparams.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)

        # transformer block weights
        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))

    def forward(self, x):
        t = x.shape[1]

        if t > self.hparams.max_len:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.hparams.max_len}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # shape (t)

        # forward the GPT model itself
        data_emb = self.data_embed(x)  # to shape (b, t, embedding_dim)
        pos_emb = self.position_embed(pos)  # to shape (t, embedding_dim)

        x = self.dropout(data_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)

        preds = self.head(x)

        return preds

    def training_step(self, batch):
        input = batch["input"]
        target = batch["target"]
        output = self(input)

        # find the mask locations
        locs = (input != target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        self.log("training_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        input = batch["input"]
        target = batch["target"]
        output = self(input)

        # find the mask locations
        locs = (input == target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        self.log("val_training_loss", loss, prog_bar=True)
        return loss


class MLP(nn.Module):
    """A two-layer MLP.

    This uses a fully-connected layer to encode the input, then applies a non-linearity,
    then uses another fully-connected layer to decode back to the initial dimension, and
    finally applies (optional) dropout.

    :param in_features: size of input layer
    :param hidden_features: size of hidden layer
    :param activation: activation function to use after the expansion; default: GELU
    :param dropout: amount of dropout
    :param bias: whether to use bias in the layers
    """

    in_features: int
    hidden_features: int
    activation: Callable
    dropout: float
    bias: bool

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        activation: Optional[Callable] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = dropout
        self.bias = bias

        self.encoder = nn.Linear(in_features, hidden_features, bias=bias)
        self.decoder = nn.Linear(hidden_features, in_features, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class SelfAttention(nn.Module):
    """Collection of self-attention heads.

    :param embedding_dim: total dimensionality of the model (equal to
        `head_size * num_heads`)
    :param num_heads: number of heads
    :param bias: whether to include bias terms
    :param dropout: amount of dropout; used both for the attention and for the residual
        pathways
    :param causal: if true, use causal self-attention
    """

    embedding_dim: int
    num_heads: int
    dropout: float
    uses_flash: bool

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool,
        dropout: float,
        bias: bool = True,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim should be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal

        # key, query, value projections for all heads, but in a batch
        self.attention = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)

        # output projection
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # regularization
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # flash attention makes GPU go brrrrr but support is only in PyTorch >= 2.0
        self.uses_flash = hasattr(F, "scaled_dot_product_attention")
        if not self.uses_flash:
            print("Using slow attention. Flash Attention requires PyTorch >= 2.0.")

            if self.causal:
                self.register_buffer("mask", torch.empty((1, 1, 0, 0), dtype=bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality
        B, T, C = x.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected input shape (..., {self.embedding_dim}, got {x.shape})"
            )

        # calculate and separate out query, key, values for all heads
        # each has shape (B, T, C)
        q, k, v = self.attention(x).split(self.embedding_dim, dim=2)

        # separate out head index and move it up next to the batch dimension
        # final shape (B, num_heads, T, head_size), where C = num_heads * head_size
        nh = self.num_heads
        hs = C // nh
        k = k.view(B, T, nh, hs).transpose(1, 2)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.uses_flash:
            # efficient attention using Flash Attention CUDA kernels
            dropout_p = self.dropout if self.training else 0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=self.causal
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

            if self.causal:
                # cache the causal mask, if we're using one
                if self.mask.shape[2] < T:
                    self.mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T) == 0
                att = att.masked_fill(self.bias[:, :, :T, :T], float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attention_dropout(att)
            # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.residual_dropout(self.projection(y))
        return y


class TransformerBlock(nn.Module):
    """A transformer block, including layer norm, self-attention, another layer norm,
    and a two-layer MLP.

    :param embedding_dim: total dimensionality of the self-attention model (equal to
        `head_size * num_heads`)
    :param num_heads: number of self-attention heads
    :param bias: whether to include bias terms; used for layernorms, attention, and MLP
    :param dropout: amount of dropout; used for attention, resiudal pathway, and MLP
    :param causal: if true, use causal self-attention
    :param mlp_expansion: ratio between embedding dimension and side of MLP hidden layer
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool,
        dropout: float,
        bias: bool = True,
        mlp_expansion: int = 4,
    ):
        super().__init__()

        self.layernorm1 = LayerNorm(embedding_dim, bias=bias)
        self.attention = SelfAttention(
            embedding_dim, num_heads, bias=bias, dropout=dropout, causal=causal
        )
        self.layernorm2 = LayerNorm(embedding_dim, bias=bias)

        hidden_dim = mlp_expansion * embedding_dim
        self.mlp = MLP(embedding_dim, hidden_dim, nn.GELU(), dropout=dropout, bias=bias)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x


class LayerNorm(nn.Module):
    """Layer normalized with optional bias.

    This is based on PyTorch's :class:`~torch.nn.LayerNorm` module but is needed because
    PyTorch's version does not support disabling the bias.

    :param shape: shape of the input, following an arbitrary number of batch dimensions;
        that is, the input has dimensions `[d1, ..., dk, shape[0], ..., shape[-1]]`
    :param eps: value added to the denominator for numerical stability
    :param bias: whether to include a bias term
    :param device: device on which to create the parameters
    :param dtype: data type to use for the parameters
    """

    normalized_shape: Tuple[int, ...]
    eps: float

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...], torch.Size],
        eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.eps = eps
        if isinstance(shape, numbers.Integral):
            self.normalized_shape = (shape,)
        else:
            self.normalized_shape = tuple(shape)

        kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(shape, **kwargs))
        self.bias = nn.Parameter(torch.empty(shape, **kwargs)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class TiedLinear(nn.Module):
    """A dense linear layer whose parameters are tied to a tensor provided by the user.

    Using this layer is equivalent to using the functional form,
    :func:`~torch.nn.functional.linear`. The utility of having a module is that it will
    show up in module summaries, which can help to make the structure of the model more
    transparent.

    :param weight: weight tensor
    :param bias: bias tensor; if not provided, there will be no bias
    """

    in_features: int
    """size of each input sample."""

    out_features: int
    """size of each output sample."""

    def __init__(
        self,
        weight: Union[torch.Tensor, nn.Parameter],
        bias: Union[None, torch.Tensor, nn.Parameter],
    ):
        super().__init__()

        if weight.ndim != 2:
            raise ValueError(
                f"weight parameter has {weight.ndim} dimensions, should have 2"
            )
        self.out_features, self.in_features = weight.shape

        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _init_by_depth(module: nn.Module, depth: int):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        std = 1 / math.sqrt(2 * fan_in * depth)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
