"""
ConvNeXt-Base Bone Fracture Model Architecture
MedGuard AI - Bone Fracture Analysis Subsystem
Genuine ConvNeXt-Base Multi-Region Radiograph Evaluation

Implements:
- torchvision.models.convnext_base with official ImageNet-pretrained weights
- 1024-dimensional pooled global feature representation
- Custom 3-stage MLP classification head: 1024 -> 512 -> 256 -> 1
- GELU hidden activations and Dropout (0.3)
- Stage-wise freezing / unfreezing controls
- Checkpoint persistence and loading
- Feature extraction interface for caching and OOD detection
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Base_Weights

from ml.tasks.fracture.config import (
    FEATURE_DIM,
    HEAD_DIMS,
    DROPOUT_RATE,
    BACKBONE_NAME
)


class ConvNeXtFractureClassifier(nn.Module):
    """
    Genuine ConvNeXt-Base with custom multi-stage classification head.
    Outputs a single raw logit for binary fracture detection.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = DROPOUT_RATE,
        feature_dim: int = FEATURE_DIM
    ):
        super().__init__()
        self.backbone_name = BACKBONE_NAME
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate

        # 1. Load official ConvNeXt-Base backbone
        weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        base_model = models.convnext_base(weights=weights)

        self.features = base_model.features
        self.avgpool = base_model.avgpool
        # Norm layer from classifier[0]
        self.norm = base_model.classifier[0]
        self.flatten = base_model.classifier[1]

        # 2. Build 3-stage MLP head: 1024 -> 512 -> 256 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

        # Initialize head weights with Kaiming / Xavier
        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled and normalized 1024-dimensional feature vector.
        Shape: (B, 1024)
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)
        normed = self.norm(pooled)
        flat = self.flatten(normed)
        return flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass returning unnormalized logit.
        Shape: (B, 1)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass returning calibrated or sigmoid probability.
        Shape: (B, 1) in [0.0, 1.0]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def freeze_backbone(self):
        """Freezes all parameters in the ConvNeXt-Base feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreezes all parameters across the entire network."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_stage(self, stage_idx: int = 7):
        """
        Selectively unfreezes the final stage (e.g. stage 7) of ConvNeXt features.
        Useful for parameter-efficient and memory-safe fine-tuning on 4 GB VRAM.
        """
        # First ensure stages up to stage_idx are frozen
        for i, block in enumerate(self.features):
            requires_grad = (i >= stage_idx)
            for param in block.parameters():
                param.requires_grad = requires_grad
        for param in self.norm.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Returns total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_pct": (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0
        }

    def save_checkpoint(
        self,
        filepath: str | Path,
        epoch: int,
        val_metric: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ):
        """Saves a resumable checkpoint atomically."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path = filepath.with_suffix(".tmp")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "best_val_metric": val_metric,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "feature_dim": self.feature_dim,
            "dropout_rate": self.dropout_rate,
            "backbone_name": self.backbone_name,
            "extra_metadata": extra_metadata or {}
        }

        torch.save(checkpoint, temp_path)
        if filepath.exists():
            filepath.unlink()
        temp_path.rename(filepath)

    def load_checkpoint(self, filepath: str | Path, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        """Loads weights and optimizer states from a checkpoint."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


def create_fracture_model(pretrained: bool = True, device: Optional[torch.device] = None) -> ConvNeXtFractureClassifier:
    """Factory function to build and place the model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtFractureClassifier(pretrained=pretrained)
    model.to(device)
    return model
