"""
TB Classification Model -- MedGuard AI
=======================================
Isolated ResNet-18 model for binary Tuberculosis (TB) screening.

Architecture:
    Backbone: ResNet-18 (ImageNet-pretrained)
    Head: Dropout(0.3) -> Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)
    Output: Single logit (pass through sigmoid for probability)

This model is intentionally kept SEPARATE from the existing ResNet18MLP
(pneumonia model) to avoid checkpoint conflicts and preserve model isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

# ── Early path bootstrap ─────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_ML_ROOT_TBM = _Path(__file__).resolve().parent.parent
for _p in (str(_ML_ROOT_TBM), str(_ML_ROOT_TBM / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
from config import MODELS_DIR

# Separate checkpoint path -- never overwrites pneumonia best_model.pth
TB_MODEL_PATH = MODELS_DIR / "best_model_tb.pth"
TB_CLASS_NAMES = ["Non-TB", "TB"]


class TBResNet18(nn.Module):
    """
    Binary TB classification model based on pretrained ResNet-18.

    Output: single logit (N, 1). Use sigmoid for probabilities.
    Loss:   BCEWithLogitsLoss with pos_weight for class imbalance.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Extract feature layers (all except final fc)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        # TB-specific classification head (separate from pneumonia model)
        self.tb_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (N, 512) feature vector from the ResNet-18 backbone."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns single binary logit (N, 1)."""
        feats = self.extract_features(x)
        return self.tb_classifier(feats)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logit, features) -- useful for Grad-CAM."""
        feats = self.extract_features(x)
        logit = self.tb_classifier(feats)
        return logit, feats

    @property
    def gradcam_layer(self) -> nn.Module:
        """Target layer for Grad-CAM visualization (last residual block)."""
        return self.layer4


def build_tb_model(pretrained: bool = True, dropout: float = 0.3,
                   device: Optional[torch.device] = None) -> TBResNet18:
    """Constructs and moves a TBResNet18 model to the target device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TBResNet18(pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TBResNet18] Total params: {n_params:,} | Trainable: {n_trainable:,} | Device: {device}")
    return model


def save_tb_checkpoint(
    model: TBResNet18,
    optimizer,
    epoch: int,
    metrics: dict,
    path: Path = TB_MODEL_PATH,
) -> None:
    """Saves model weights, optimizer state, epoch, and validation metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }, path)
    print(f"[TBResNet18] Checkpoint saved: {path} (epoch={epoch})")


def load_tb_checkpoint(
    path: Path = TB_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[TBResNet18, dict]:
    """
    Loads the best TB model checkpoint.

    Returns:
        model:   TBResNet18 loaded with saved weights.
        meta:    dict with 'epoch' and 'metrics'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not path.exists():
        raise FileNotFoundError(f"TB checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = TBResNet18(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[TBResNet18] Loaded checkpoint: epoch={ckpt.get('epoch')}, metrics={ckpt.get('metrics')}")
    return model, {"epoch": ckpt.get("epoch"), "metrics": ckpt.get("metrics")}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = build_tb_model(pretrained=True, device=device)
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape: {logits.shape}")   # expected: (2, 1)
    probs = torch.sigmoid(logits)
    print(f"Probabilities: {probs.squeeze().tolist()}")
