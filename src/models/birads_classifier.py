"""
BI-RADS classifier built on EfficientNet-B4 (transfer learning).

EfficientNet-B4 is a strong baseline for medical image classification:
  - Good accuracy / parameter tradeoff
  - Works well with small datasets via transfer learning
  - torchvision provides pretrained weights
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights


def build_model(num_classes: int, pretrained: bool = True,
                dropout: float = 0.4) -> nn.Module:
    """
    Return an EfficientNet-B4 with a custom classification head.

    Args:
        num_classes: 2 for binary, 8 for full multiclass
        pretrained:  use ImageNet pretrained weights
        dropout:     dropout rate before the final FC layer
    """
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class BiRadsClassifier(nn.Module):
    """
    Thin wrapper so we can add forward-hooks / additional heads later.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        self.backbone = build_model(num_classes, pretrained, dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
