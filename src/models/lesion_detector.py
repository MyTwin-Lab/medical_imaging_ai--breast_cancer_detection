"""
Lesion detector built on Faster R-CNN with ResNet-50 FPN backbone.

Pretrained on COCO (91 classes) → fine-tuned for mammography lesions:
  0: background
  1: Mass
  2: Calcification
  3: Asymmetry
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Return Faster R-CNN with a custom box-predictor head.

    Args:
        num_classes: number of classes INCLUDING background (default 4)
        pretrained:  use COCO pretrained backbone weights
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class LesionDetector(nn.Module):
    """Thin wrapper around Faster R-CNN for mammography lesion detection."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.model = build_detector(num_classes, pretrained)
        self.num_classes = num_classes

    def forward(self, images, targets=None):
        """
        Training:   forward(images, targets) → dict of losses
        Inference:  forward(images)          → list of prediction dicts
        """
        return self.model(images, targets)
