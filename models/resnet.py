# mocomotion/models/resnet.py

from typing import Tuple

import torch.nn as nn
from torchvision import models as tv


SUPPORTED_BACKBONES = {"resnet18", "resnet34", "resnet50"}


def build_resnet(backbone_name: str = "resnet50") -> Tuple[nn.Module, int]:
    """
    Build a randomly initialized ResNet backbone.

    Returns:
        backbone: nn.Module that outputs global pooled features
        feat_dim: feature dimension (e.g. 512 for resnet18/34, 2048 for resnet50)

    IMPORTANT:
    - weights=None ensures we do NOT load ImageNet pretrained weights
      (course requires random initialization).
    """
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Supported: {SUPPORTED_BACKBONES}"
        )

    resnet_fn = getattr(tv, backbone_name)
    backbone = resnet_fn(weights=None)  # random init, no pretraining

    # The original fc is classification head; we want pure features.
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()  # so forward() returns the pooled features

    return backbone, feat_dim


if __name__ == "__main__":
    # Quick local sanity check
    import torch

    model, dim = build_resnet("resnet50")
    x = torch.randn(2, 3, 96, 96)  # dummy 96x96 batch
    with torch.no_grad():
        feat = model(x)
    print("feat shape:", feat.shape)  # expect [2, feat_dim]
    print("feat_dim:", dim)
