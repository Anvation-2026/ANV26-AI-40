from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Path bootstrap
from src import _ML_ROOT  # noqa: F401

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


def build_model(pretrained: bool = True) -> nn.Module:
    """
    Builds ResNet-18 with 2-class classification head.
    If pretrained=True, downloads and initializes with ImageNet weights.
    If pretrained=False (inference mode), initializes without pre-trained weights.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    training_config: Dict[str, Any],
    best_epoch: int,
    best_val_metric: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    models_dir: Path = MODELS_DIR,
) -> Path:
    """
    Saves best_model.pth with full required metadata.
    Also preserves previous checkpoint as best_model.prev.pth if it exists.
    Updates model_card.json and artifacts_manifest.json.
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

    # Prepare checkpoint payload
    checkpoint_data = {
        "state_dict": model.state_dict(),
        "arch": "resnet18",
        "num_classes": len(CLASS_NAMES),
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

    # Save model_card.json
    model_card = {
        "model_name": "MedGuard AI ResNet-18 Chest X-Ray Classifier",
        "model_version": full_model_version,
        "base_model": "torchvision.models.resnet18",
        "task": "Binary Classification (Normal vs Pneumonia)",
        "intended_use": "Educational decision-support research prototype. Not for clinical diagnosis.",
        "created_utc": checkpoint_data["created_utc"],
        "architecture": {
            "backbone": "ResNet-18",
            "feature_dim": 512,
            "classifier": "Linear(512, 2)",
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

    # Update artifacts_manifest.json
    update_artifacts_manifest(models_dir, full_model_version)
    print(f"[Model] Checkpoint and model card saved: {path} (version: {full_model_version})")
    return path


def update_artifacts_manifest(models_dir: Path = MODELS_DIR, model_version: Optional[str] = None) -> Path:
    """Scans models_dir, computes SHA-256 for all present files, and saves artifacts_manifest.json."""
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
    """
    Loads model checkpoint safely with weights_only handling.
    Returns (model, metadata_dict).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    # Build blank architecture
    model = build_model(pretrained=False)

    try:
        # PyTorch >= 2.4 default weights_only=True
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for earlier torch or complex dicts
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format in {path}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Calculate model_version string if sha is present
    ckpt_sha256 = compute_file_sha256(path)
    base_ver = checkpoint.get("model_version", MODEL_VERSION_BASE)
    checkpoint["full_model_version"] = f"{base_ver}+{ckpt_sha256[:8]}"
    checkpoint["sha256"] = ckpt_sha256

    return model, checkpoint
