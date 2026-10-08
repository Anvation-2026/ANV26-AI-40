from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_VERSION_BASE,
    MODELS_DIR,
)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


class ResNet18MLP(nn.Module):
    """
    ImageNet-pretrained ResNet-18 feature extractor + Multi-Layer Perceptron (MLP) head.
    Architecture:
      Backbone: ResNet-18 (conv1 through avgpool -> 512-dim features)
      Classifier:
        - Linear(512, 256)
        - BatchNorm1d(256)
        - ReLU
        - Dropout(0.3)
        - Linear(256, 64)
        - ReLU
        - Dropout(0.2)
        - Linear(64, 1)  -> Single binary classification logit
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Retain all feature extraction stages
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # MLP classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Runs convolutional backbone to produce 512-d feature vector."""
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
        return self.classifier(feats)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns binary logit (N, 1) and penultimate features (N, 512)."""
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats

    def get_2d_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps binary logit z to 2-class logits [-z/2, z/2] so that
        softmax([-z/2, z/2]) produces [1 - sigmoid(z), sigmoid(z)].
        Provides backward compatibility with 2-class downstream tools.
        """
        logit = self.forward(x)  # (N, 1)
        half_z = logit / 2.0
        return torch.cat([-half_z, half_z], dim=1)  # (N, 2)


class ResNet18Linear(nn.Module):
    """
    Baseline ResNet-18 with 2-class linear classification head.
    Matches original baseline architecture.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, len(CLASS_NAMES))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
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
        feats = self.extract_features(x)
        return self.fc(feats)

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.extract_features(x)
        logits = self.fc(feats)
        return logits, feats

    def get_2d_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def build_model(pretrained: bool = True, arch: str = "resnet18_mlp") -> nn.Module:
    """
    Builds either ResNet-18 + MLP (arch='resnet18_mlp')
    or baseline ResNet-18 Linear (arch='resnet18_linear' or 'resnet18').
    """
    if "mlp" in arch.lower():
        return ResNet18MLP(pretrained=pretrained)
    return ResNet18Linear(pretrained=pretrained)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    training_config: Dict[str, Any],
    best_epoch: int,
    best_val_metric: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    models_dir: Path = MODELS_DIR,
    arch: str = "resnet18_mlp",
) -> Path:
    """
    Saves checkpoint with complete metadata and backup of previous model.
    """
    path = Path(path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if path.exists():
        prev_path = models_dir / "best_model.prev.pth"
        try:
            if prev_path.exists():
                prev_path.unlink()
            path.rename(prev_path)
            print(f"[Model] Preserved previous model at {prev_path.name}")
        except Exception as e:
            print(f"[Model] Warning: could not backup previous checkpoint: {e}")

    # Determine architecture metadata
    is_mlp = isinstance(model, ResNet18MLP) or "mlp" in arch.lower()
    arch_name = "resnet18_mlp" if is_mlp else "resnet18_linear"
    classifier_desc = (
        "Linear(512, 256) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> "
        "Linear(256, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 1)"
        if is_mlp
        else "Linear(512, 2)"
    )

    checkpoint_data = {
        "state_dict": model.state_dict(),
        "arch": arch_name,
        "num_classes": 1 if is_mlp else len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "input_size": IMAGE_SIZE,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "training_config": training_config,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "dataset": dataset_meta,
        "torch_version": torch.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION_BASE,
    }

    torch.save(checkpoint_data, path)
    ckpt_sha256 = compute_file_sha256(path)
    full_model_version = f"{MODEL_VERSION_BASE}+{ckpt_sha256[:8]}"

    model_card = {
        "model_name": f"MedGuard AI {arch_name.upper()} Chest X-Ray Classifier",
        "model_version": full_model_version,
        "base_model": "torchvision.models.resnet18",
        "task": "Binary Classification (Normal vs Pneumonia)",
        "intended_use": "Educational decision-support research prototype. Not for clinical diagnosis.",
        "created_utc": checkpoint_data["created_utc"],
        "architecture": {
            "backbone": "ResNet-18",
            "feature_dim": 512,
            "classifier": classifier_desc,
        },
        "input": {
            "format": "Grayscale chest X-ray replicated to 3 channels",
            "size": [IMAGE_SIZE, IMAGE_SIZE],
            "normalization": checkpoint_data["normalization"],
        },
        "training": {
            "config": training_config,
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
        },
        "dataset": dataset_meta,
        "limitations": [
            "Trained exclusively on pediatric chest X-rays (PneumoniaMNIST).",
            "Not validated for adult populations or cross-scanner domain shifts.",
            "Educational research prototype only; no clinical decision claims.",
        ],
    }

    model_card_path = models_dir / "model_card.json"
    with open(model_card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    update_artifacts_manifest(models_dir, full_model_version)
    print(f"[Model] Checkpoint and model card saved: {path} (version: {full_model_version})")
    return path


def update_artifacts_manifest(models_dir: Path = MODELS_DIR, model_version: Optional[str] = None) -> Path:
    models_dir = Path(models_dir)
    manifest_path = models_dir / "artifacts_manifest.json"

    artifacts = {}
    best_model_path = models_dir / "best_model.pth"

    for file_path in sorted(models_dir.iterdir()):
        if file_path.is_file() and file_path.name != "artifacts_manifest.json":
            artifacts[file_path.name] = {
                "sha256": compute_file_sha256(file_path),
                "size_bytes": file_path.stat().st_size,
            }

    if model_version is None:
        if best_model_path.exists():
            sha = artifacts.get("best_model.pth", {}).get("sha256", "unknown")
            model_version = f"{MODEL_VERSION_BASE}+{sha[:8]}"
        else:
            model_version = f"{MODEL_VERSION_BASE}+unavailable"

    manifest_data = {
        "model_version": model_version,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_path


def load_checkpoint(
    path: Path,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format in {path}")

    arch = checkpoint.get("arch", "resnet18")
    model = build_model(pretrained=False, arch=arch)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    ckpt_sha256 = compute_file_sha256(path)
    base_ver = checkpoint.get("model_version", MODEL_VERSION_BASE)
    checkpoint["full_model_version"] = f"{base_ver}+{ckpt_sha256[:8]}"
    checkpoint["sha256"] = ckpt_sha256

    return model, checkpoint
