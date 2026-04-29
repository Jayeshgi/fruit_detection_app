"""
fruit_model.py
──────────────────────────────────────────────────────────────────────────────
Defines the FruitClassifier model using Transfer Learning.

Architecture:
  - Backbone: MobileNetV3-Small (pretrained on ImageNet)
  - Head:     Custom classifier layer matching our number of fruit classes

Why MobileNetV3?
  - Extremely fast inference on CPU (important for local development)
  - Small model size (~2.5 MB weights)
  - Still achieves high accuracy with transfer learning on clean datasets
──────────────────────────────────────────────────────────────────────────────
"""
import torch
import torch.nn as nn
from torchvision import models

from config import MODEL_NAME, FREEZE_BACKBONE


def create_model(num_classes: int, freeze_backbone: bool = FREEZE_BACKBONE):
    """
    Creates a fruit classification model using transfer learning.

    Args:
        num_classes:     Number of fruit classes to predict.
        freeze_backbone: If True, freeze all pretrained layers so only
                         the final classifier is trained (faster, needs
                         less data). Set to False for fine-tuning.

    Returns:
        model: nn.Module ready for training
    """
    if MODEL_NAME == "mobilenet_v3_small":
        # ── MobileNetV3-Small ─────────────────────────────────────────────
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        # Replace the classifier head
        # Original: Linear(576, 1000)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif MODEL_NAME == "mobilenet_v3_large":
        # ── MobileNetV3-Large ─────────────────────────────────────────────
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)

        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif MODEL_NAME == "resnet18":
        # ── ResNet18 ──────────────────────────────────────────────────────
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model: {MODEL_NAME}. "
            f"Choose from: mobilenet_v3_small, mobilenet_v3_large, resnet18"
        )

    # Count trainable vs frozen parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n[Model] {MODEL_NAME}")
    print(f"  Total parameters      : {total_params:,}")
    print(f"  Trainable parameters  : {trainable_params:,}")
    print(f"  Frozen parameters     : {frozen_params:,}")
    print(f"  Output classes        : {num_classes}")

    return model
