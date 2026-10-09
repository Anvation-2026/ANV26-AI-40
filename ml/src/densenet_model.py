"""
ml/src/densenet_model.py -- DenseNet-201 Architecture for 14-label Multi-Label Chest Radiograph Classification.
=============================================================================================================
Architecture:
  - Backbone: ImageNet-pretrained DenseNet-201 (1920 features)
  - Penultimate Feature Dimension: 1920
  - MLP Classifier Head: 1920 -> 512 -> 256 -> 14
  - Hidden Activation: ReLU
  - Dropout: 0.3
  - Output: 14 independent logits
  - Inference: Sigmoid activation per disease class
  - Grad-CAM Target: features.denseblock4 (or features.norm5)

Preserves existing ResNet-18 binary model and checkpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import DenseNet201_Weights, densenet201

# Path bootstrap
try:
    from config import MODELS_DIR
except ModuleNotFoundError:
    from ..config import MODELS_DIR

# Official MedMNIST ChestMNIST 14 disease labels (in strict canonical order)
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
DENSENET_FEATURE_DIM = 1920
DENSENET_MODELS_DIR = MODELS_DIR / "densenet201"
DENSENET_CHECKPOINT_PATH = DENSENET_MODELS_DIR / "best_model.pth"


class DenseNet201Chest14(nn.Module):
    """
    Upgraded DenseNet-201 multi-label chest radiograph model.
    Feature Extractor: Pretrained ImageNet DenseNet-201 (1920-d features).
    Classifier: MLP Head (1920 -> 512 -> 256 -> 14) with ReLU and Dropout(0.3).
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet201(weights=weights)

        # Feature extractor stages
        self.features = backbone.features

        # Hidden activation
        act_layer = nn.ReLU(inplace=True) if activation.lower() == "relu" else nn.GELU()

        # Multi-layer Perceptron (MLP) Classifier: 1920 -> 512 -> 256 -> 14
        self.classifier = nn.Sequential(
            nn.Linear(DENSENET_FEATURE_DIM, 512),
            act_layer,
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            act_layer,
            nn.Dropout(p=dropout),
            nn.Linear(256, NUM_CHEST_CLASSES),
        )

        self.class_names = CHESTMNIST_14_LABELS
        self.num_classes = NUM_CHEST_CLASSES
        self.feature_dim = DENSENET_FEATURE_DIM

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1920-dimensional global pooled feature vector from backbone.
        Output shape: (N, 1920).
        """
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing 14 binary classification logits.
        Output shape: (N, 14).
        """
        feats = self.extract_features(x)
        return self.classifier(feats)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns logits (N, 14) and penultimate feature representation (N, 1920).
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-class independent sigmoid probabilities in [0, 1].
        Shape: (N, 14).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Stage 1 transfer learning:
        Freezes the entire DenseNet-201 feature extractor to train only the MLP classifier head.
        Saves significant VRAM and compute on 4 GB GPUs during initial warmup.
        """
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def unfreeze_denseblock4(self) -> None:
        """
        Stage 2 fine-tuning:
        Unfreezes denseblock4 and final norm5 while keeping denseblock1-3 frozen.
        Enables high-level visual representation fine-tuning within 4 GB VRAM constraints.
        """
        # Ensure early layers stay frozen
        self.freeze_backbone(True)
        # Unfreeze denseblock4 and norm5
        for param in self.features.denseblock4.parameters():
            param.requires_grad = True
        for param in self.features.norm5.parameters():
            param.requires_grad = True

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the target layer for Grad-CAM visualization."""
        return self.features.denseblock4


def build_densenet201(
    pretrained: bool = True,
    dropout: float = 0.3,
    activation: str = "relu",
    device: Optional[torch.device] = None,
) -> DenseNet201Chest14:
    """
    Factory function for DenseNet-201 with 1920 -> 512 -> 256 -> 14 MLP head.
    """
    model = DenseNet201Chest14(pretrained=pretrained, dropout=dropout, activation=activation)
    if device is not None:
        model = model.to(device)
    return model


def save_densenet_checkpoint(
    model: DenseNet201Chest14,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    stage: str,
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, float]],
    save_path: Union[str, Path] = DENSENET_CHECKPOINT_PATH,
) -> Path:
    """
    Saves a complete resumable checkpoint for DenseNet-201.
    Never touches ResNet-18 checkpoints.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "stage": stage,
        "metrics": metrics,
        "thresholds": thresholds or {},
        "class_names": CHESTMNIST_14_LABELS,
        "num_classes": NUM_CHEST_CLASSES,
        "feature_dim": DENSENET_FEATURE_DIM,
        "arch": "densenet201_mlp_1920_512_256_14",
        "dataset": "ChestMNIST 224x224 (NIH-ChestXray14)",
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(state, save_path)
    return save_path


def load_densenet_checkpoint(
    checkpoint_path: Union[str, Path] = DENSENET_CHECKPOINT_PATH,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[DenseNet201Chest14, Dict[str, Any]]:
    """
    Loads DenseNet-201 model from a checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_densenet201(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, ckpt
