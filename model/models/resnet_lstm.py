"""ResNet-LSTM — the paper's headline architecture (Keras Fig. 1 translated to PyTorch).

1-D CNN feature extractor (Conv→BN→ReLU ×3) with a residual skip around the last conv, dropout,
then an LSTM over the time axis, then a dense classification head. Input (B, window, n_features);
output logits (B, n_classes) — use with nn.CrossEntropyLoss.

Conv1d in torch is (B, C, L), so we transpose (B,W,F)->(B,F,W) for the convs (channels = features,
length = time) and back to (B,W,C) for the LSTM.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResNetLSTM(nn.Module):
    def __init__(self, n_features: int, window: int = 96, n_classes: int = 2,
                 filters: int = 64, kernel: int = 3, lstm_units: int = 64, dropout: float = 0.3):
        super().__init__()
        self.c1 = nn.Conv1d(n_features, filters, kernel, padding="same")
        self.b1 = nn.BatchNorm1d(filters)
        self.c2 = nn.Conv1d(filters, filters, kernel, padding="same")
        self.b2 = nn.BatchNorm1d(filters)
        self.c3 = nn.Conv1d(filters, filters, kernel, padding="same")   # residual block conv
        self.b3 = nn.BatchNorm1d(filters)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(filters, lstm_units, batch_first=True)
        self.head = nn.Linear(lstm_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, W, F)
        x = x.transpose(1, 2)                              # (B, F, W)
        x = torch.relu(self.b1(self.c1(x)))
        x = torch.relu(self.b2(self.c2(x)))
        skip = x
        y = self.b3(self.c3(x))
        y = torch.relu(skip + y)                           # residual add
        y = self.drop(y).transpose(1, 2)                   # (B, W, filters)
        out, _ = self.lstm(y)
        return self.head(out[:, -1])                       # logits at last timestep
