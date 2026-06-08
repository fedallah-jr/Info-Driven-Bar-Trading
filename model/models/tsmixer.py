"""TSMixer — all-MLP time-series mixer (the paper's MLP-Mixer baseline), PyTorch.

Stacks residual blocks of (time-mixing MLP) + (feature-mixing MLP), then global-average-pools over
time into a classification head. Input (B, window, n_features) -> logits (B, n_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MixerBlock(nn.Module):
    def __init__(self, window: int, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_features)
        self.time_mlp = nn.Sequential(nn.Linear(window, window), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(window, window))
        self.ln2 = nn.LayerNorm(n_features)
        self.feat_mlp = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B, W, F)
        z = self.ln1(x).transpose(1, 2)                    # (B, F, W) — mix across time
        x = x + self.time_mlp(z).transpose(1, 2)
        x = x + self.feat_mlp(self.ln2(x))                 # mix across features
        return x


class TSMixer(nn.Module):
    def __init__(self, n_features: int, window: int = 96, n_classes: int = 2,
                 blocks: int = 4, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.ModuleList([_MixerBlock(window, n_features, hidden, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(n_features)
        self.head = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B, W, F)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))         # global avg pool over time
