"""Vanilla Transformer encoder for sequence classification (PyTorch).

Project features -> d_model, add a learnable positional embedding, run a few self-attention encoder
layers, mean-pool over time, classify. Input (B, window, n_features) -> logits (B, n_classes).
A different inductive bias from the conv/LSTM and MLP-mixer models, so it trades differently.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, n_features: int, window: int = 96, n_classes: int = 2,
                 d_model: int = 64, nhead: int = 4, layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, F)
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(self.norm(x).mean(dim=1))         # mean-pool over time -> logits
