"""PyTorch transformer model for hybrid backward pass.

A lightweight pre-norm transformer used as the autograd mirror for the
wgpu GPU model. Architecture is identical to TransformerWGPU so weights
can be synced between the two.

Architecture:
    Token Embedding (vocab_size -> d_model) + Sinusoidal Position Encoding
    -> N x TransformerLayer (pre-norm, multi-head self-attention, FFN)
    -> Final LayerNorm
    -> MLM Head: Linear -> GELU -> LayerNorm -> Linear(vocab_size)
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Generate sinusoidal position encodings."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.from_numpy(pe)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.W_q(x).reshape(B, L, H, D).permute(0, 2, 1, 3)
        k = self.W_k(x).reshape(B, L, H, D).permute(0, 2, 1, 3)
        v = self.W_v(x).reshape(B, L, H, D).permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.W_o(out)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with GELU activation."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class MLMHead(nn.Module):
    """Masked language model head: Linear -> GELU -> LayerNorm -> Linear."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense(x))
        x = self.norm(x)
        return self.proj(x)


class Transformer(nn.Module):
    """Lightweight pre-norm transformer with MLM head (PyTorch).

    Args:
        d_model: Hidden dimension (default 128).
        n_layers: Number of transformer layers (default 6).
        n_heads: Number of attention heads (default 4).
        d_ff: Feed-forward dimension (default 512).
        max_len: Maximum sequence length (default 1024).
        dropout: Dropout rate (default 0.1).
        vocab_size: Token vocabulary size (default 256).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 512,
        max_len: int = 1024,
        dropout: float = 0.1,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "_pos_enc", _sinusoidal_position_encoding(max_len, d_model)
        )
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.mlm_head = MLMHead(d_model, vocab_size)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        tok_emb = self.token_embedding(token_ids)
        pos_emb = self._pos_enc[:L]
        return self.embed_dropout(tok_emb + pos_emb)

    def _make_padding_mask(self, token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        pad_mask = (token_ids == pad_id).float()
        return pad_mask[:, None, None, :] * -1e9

    def forward_mlm(self, token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Forward pass returning MLM logits (B, L, vocab_size)."""
        x = self._embed(token_ids)
        mask = self._make_padding_mask(token_ids, pad_id)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        return self.mlm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
