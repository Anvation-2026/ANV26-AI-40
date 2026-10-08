"""
ml/src/densenet_model.py — DenseNet-201 Architecture for 14-label Multi-Label Chest X-ray Classification.

Trained on MedMNIST ChestMNIST 224x224 (NIH-ChestXray14 subset).
Preserves existing ResNet-18 binary model and checkpoints.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import DenseNet201_Weights, densenet201

CHESTMNIST_14_LABELS: List[str] = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
]

NUM_CHEST_CLASSES = len(CHESTMNIST_14_LABELS)  # 14


class DenseNet201Chest14(nn.Module):
    """
    DenseNet-201 multi-label chest radiograph classification model.
    Feature extractor: Pretrained ImageNet DenseNet-201 (1920 features).
    Classification head: Multi-label linear layer producing 14 independent binary logits.
    """

    def __init__(self, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet201(weights=weights)

        # Feature extractor stages
        self.features = backbone.features

        # Memory-conscious classifier head
        # 1920 in-features from DenseNet-201 top transition/dense block
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(1920, NUM_CHEST_CLASSES),
        )

        self.class_names = CHESTMNIST_14_LABELS

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 1920-dimensional global pooled feature vector."""
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing 14 binary classification logits.
        Output shape: (N, 14)
        """
        feats = self.extract_features(x)
        return self.classifier(feats)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits (N, 14) and penultimate feature representation (N, 1920)."""
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-class sigmoid probabilities in [0, 1]. Shape: (N, 14)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def freeze_early_layers(self, freeze: bool = True) -> None:
        """
        Memory-conscious transfer learning:
        Freezes conv0, denseblock1, denseblock2, denseblock3 to save backprop memory
        on 4 GB VRAM GPUs during warmup training.
        """
        early_modules = [
            self.features.conv0,
            self.features.denseblock1,
            self.features.transition1,
            self.features.denseblock2,
            self.features.transition2,
            self.features.denseblock3,
            self.features.transition3,
        ]
        for module in early_modules:
            for param in module.parameters():
                param.requires_grad = not freeze


def build_densenet201(pretrained: bool = True, drop_rate: float = 0.2) -> DenseNet201Chest14:
    """Factory function for DenseNet201Chest14."""
    return DenseNet201Chest14(pretrained=pretrained, drop_rate=drop_rate)
