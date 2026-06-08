"""Model registry — name -> builder, so train.py is model-agnostic.

Every builder has the signature build(n_features, window, n_classes, **hparams) -> nn.Module,
taking (B, window, n_features) and returning logits (B, n_classes).
"""
from __future__ import annotations

REGISTRY = ("resnet_lstm", "tsmixer", "transformer")


def build_model(name: str, n_features: int, window: int = 96, n_classes: int = 2, **hp):
    if name == "resnet_lstm":
        from models.resnet_lstm import ResNetLSTM
        return ResNetLSTM(n_features, window, n_classes, **hp)
    if name == "tsmixer":
        from models.tsmixer import TSMixer
        return TSMixer(n_features, window, n_classes, **hp)
    if name == "transformer":
        raise NotImplementedError("transformer: add models/transformer.py (vanilla encoder)")
    raise ValueError(f"unknown model '{name}' (registry: {REGISTRY})")
