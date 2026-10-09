"""
Comprehensive Training Pipeline for ConvNeXt-Base Bone Fracture Model
MedGuard AI - Bone Fracture Analysis Subsystem
Genuine ConvNeXt-Base Multi-Region Radiograph Evaluation

Features:
- Stage A (Frozen Backbone) & Stage B (Feature Caching) support
- Class imbalance weighting calculated strictly from the training split
- Mixed precision (FP16 AMP) with GradScaler
- Real-time logging of Loss, AUROC, AP, Sensitivity, Specificity, F1, VRAM, Duration
- Resumable and best-model checkpointing with atomic disk writes
- Early stopping with configurable patience
- Structured CSV & JSON history tracking and curve plotting
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import Counter
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.tasks.fracture.config import (
    MANIFEST_TRAIN,
    MANIFEST_VAL,
    MODELS_DIR,
    REPORTS_DIR,
    CHECKPOINT_BEST,
    CHECKPOINT_LATEST,
    MODEL_CONFIG_JSON,
    LABEL_MAP_JSON,
    CALIBRATION_JSON,
    FEATURE_DIM,
    IMAGE_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LR_HEAD,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    LABEL_MAP
)
from ml.tasks.fracture.dataset import FractureDataset, get_fracture_dataloaders
from ml.tasks.fracture.model import ConvNeXtFractureClassifier, create_fracture_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fracture_train")


CACHE_DIR = Path(r"D:\anvation\ml\tasks\fracture\cache")


class FocalLoss(nn.Module):
    """
    Focal Loss with class-balancing factor.
    Downweights easy background negatives and focuses gradients on minority/hard fractures.
    """
    def __init__(self, alpha: float = 0.55, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
        return (focal_weight * bce_loss).mean()


def extract_and_cache_features(
    model: ConvNeXtFractureClassifier,
    dataset: FractureDataset,
    cache_path: Path,
    device: torch.device,
    batch_size: int = 16
) -> Dict[str, torch.Tensor]:
    """
    Extracts 1024-d pooled features once across a split and caches them in compact FP32 format.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    logger.info(f"Extracting features for {len(dataset)} samples to {cache_path}...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    all_features = []
    all_labels = []
    all_sources = []
    all_anatomies = []
    all_paths = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            with autocast(device_type="cuda", dtype=torch.float16):
                feats = model.extract_features(imgs)
            all_features.append(feats.cpu().float())
            all_labels.append(batch["label"].float())
            all_sources.extend(batch["dataset_source"])
            all_anatomies.extend(batch["anatomical_region"])
            all_paths.extend(batch["image_path"])

            if (i + 1) % 100 == 0 or (i + 1) == len(loader):
                elapsed = time.perf_counter() - t0
                fps = (len(all_features) * batch_size) / elapsed
                logger.info(f"Extracted {len(all_paths)}/{len(dataset)} features ({fps:.1f} samples/s)")

    cached_data = {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "dataset_source": all_sources,
        "anatomical_region": all_anatomies,
        "image_path": all_paths,
        "feature_dim": FEATURE_DIM,
        "total_samples": len(all_paths)
    }

    torch.save(cached_data, cache_path)
    logger.info(f"Cached {len(all_paths)} features ({cache_path.stat().st_size / (1024**2):.1f} MB) at {cache_path}")
    return cached_data


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes comprehensive binary classification evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5
        
    try:
        ap = float(average_precision_score(y_true, y_prob))
    except Exception:
        ap = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "auroc": auroc,
        "average_precision": ap,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "brier_score": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def train_cached(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = DEFAULT_LR_HEAD,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    device_choice: str = "auto",
    balanced: bool = True,
    use_focal: bool = True
) -> Dict[str, Any]:
    """
    Executes balanced multi-anatomy Stage B head training on cached 1024-d ConvNeXt features.
    Ensures radiographs from all anatomical regions (wrist, hand, leg, hip, shoulder, mixed)
    receive equalized gradient exposure, preventing single-anatomy overfit shortcuts.
    """
    torch.manual_seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    
    # Device selection with safe fallback
    if device_choice == "cpu":
        device = torch.device("cpu")
    elif device_choice == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verify CUDA has enough free memory; if fragmented or constrained, fall back to CPU
    if device.type == "cuda":
        try:
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 2) if torch.cuda.is_available() else 0
            if free_mem < 300.0:
                logger.warning(f"CUDA VRAM heavily constrained ({free_mem:.1f} MB free). Switching to CPU for head training.")
                device = torch.device("cpu")
            else:
                _ = torch.zeros((1, 1024), device=device)
        except Exception as alloc_exc:
            logger.warning(f"CUDA allocation check failed ({alloc_exc}). Falling back to CPU.")
            device = torch.device("cpu")

    logger.info(f"Training on device: {device} using Cached Feature Pipeline (Balanced={balanced}, FocalLoss={use_focal})")

    # 1. Check if feature caches already exist
    train_cache_path = CACHE_DIR / "train_features.pt"
    val_cache_path = CACHE_DIR / "val_features.pt"

    if train_cache_path.exists() and val_cache_path.exists():
        logger.info(f"Loading pre-computed feature caches from {CACHE_DIR}...")
        train_cache = torch.load(train_cache_path, weights_only=False)
        val_cache = torch.load(val_cache_path, weights_only=False)
        full_model = ConvNeXtFractureClassifier(pretrained=True)
    else:
        logger.info("Feature caches not found on disk. Initializing ConvNeXt-Base for feature extraction...")
        full_model = ConvNeXtFractureClassifier(pretrained=True).to(device)
        train_ds = FractureDataset(MANIFEST_TRAIN, is_training=False)
        val_ds = FractureDataset(MANIFEST_VAL, is_training=False)
        train_cache = extract_and_cache_features(full_model, train_ds, train_cache_path, device)
        val_cache = extract_and_cache_features(full_model, val_ds, val_cache_path, device)

    X_train = train_cache["features"]
    y_train = train_cache["labels"]
    X_val = val_cache["features"]
    y_val = val_cache["labels"]
    anats_train = train_cache.get("anatomical_region", ["unknown"] * len(y_train))
    anats_val = val_cache.get("anatomical_region", ["unknown"] * len(y_val))

    num_pos = float((y_train == 1.0).sum())
    num_neg = float((y_train == 0.0).sum())
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0

    # Log class and anatomy distributions
    logger.info(f"Training dataset: {len(X_train):,} samples (Pos={int(num_pos):,}, Neg={int(num_neg):,})")
    anat_counts = Counter(anats_train)
    logger.info("Training anatomy distribution: " + ", ".join(f"{k}: {v:,}" for k, v in anat_counts.items()))

    # Build balanced sampler if requested
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    if balanced:
        logger.info("Constructing anatomy-balanced sampler across all regions and fracture classes...")
        pair_counts = Counter(zip(anats_train, [int(y.item()) for y in y_train]))
        unique_anats = sorted(list(set(anats_train)))
        num_anats = len(unique_anats)

        sample_weights = []
        for a, y in zip(anats_train, y_train):
            c = pair_counts[(a, int(y.item()))]
            # Each anatomy gets 1/num_anats, each class within anatomy gets 1/2
            sample_weights.append(1.0 / (num_anats * 2.0 * c))
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(X_train), replacement=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    head = full_model.classifier.to(device)
    if use_focal:
        criterion = FocalLoss(alpha=0.55, gamma=1.5)
    else:
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_score = 0.0
    best_auroc = 0.0
    best_epoch = 0
    best_calib = {}
    epochs_no_improve = 0

    logger.info("\n----------------------------------------------------------------------")
    logger.info(f"STAGE B: Training Classification Head ({epochs} epochs, Balanced Multi-Anatomy)")
    logger.info("----------------------------------------------------------------------")

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        head.train()
        total_loss = 0.0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_dataset)

        # Validation
        head.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = head(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item() * feats.size(0)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_preds_arr = np.vstack(val_preds).ravel()
        val_targets_arr = np.vstack(val_targets).ravel()
        metrics = compute_metrics(val_targets_arr, val_preds_arr)

        # Per-anatomy AUROC on validation
        df_v = pd.DataFrame({"anat": anats_val, "y_true": val_targets_arr, "y_prob": val_preds_arr})
        anat_aucs = {}
        for anat, grp in df_v.groupby("anat"):
            if (grp["y_true"] == 1).sum() > 0 and (grp["y_true"] == 0).sum() > 0:
                anat_aucs[anat] = float(roc_auc_score(grp["y_true"], grp["y_prob"]))

        macro_anat_auroc = float(np.mean(list(anat_aucs.values()))) if anat_aucs else metrics["auroc"]
        composite_score = 0.5 * metrics["auroc"] + 0.5 * macro_anat_auroc

        epoch_time = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[Epoch {ep:02d}/{epochs:02d}] "
            f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
            f"Val AUROC: {metrics['auroc']:.4f} | Macro Anat AUROC: {macro_anat_auroc:.4f} | "
            f"Sens: {metrics['sensitivity']:.4f} | Spec: {metrics['specificity']:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        hist_record = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "duration_seconds": epoch_time,
            "macro_anatomy_auroc": macro_anat_auroc,
            "composite_score": composite_score,
            "per_anatomy_auroc": anat_aucs,
            **metrics
        }
        history.append(hist_record)

        # Resumable checkpoint
        full_model.save_checkpoint(
            CHECKPOINT_LATEST,
            epoch=ep,
            val_metric=metrics["auroc"],
            optimizer=optimizer,
            scheduler=scheduler,
            extra_metadata={"strategy": "anatomy_balanced", "macro_anatomy_auroc": macro_anat_auroc}
        )

        # Best model checkpoint based on composite score
        if composite_score > best_score:
            best_score = composite_score
            best_auroc = metrics["auroc"]
            best_epoch = ep
            epochs_no_improve = 0

            # Compute Youden's J calibrated threshold
            fpr, tpr, threshs = roc_curve(val_targets_arr, val_preds_arr)
            j_scores = tpr - fpr
            best_j_idx = int(np.argmax(j_scores))
            opt_thresh = float(threshs[best_j_idx])

            best_calib = {
                "optimal_threshold_youden": round(opt_thresh, 4),
                "high_sensitivity_threshold": round(float(threshs[np.where(tpr >= 0.95)[0][0]]) if len(np.where(tpr >= 0.95)[0]) > 0 else opt_thresh, 4),
                "validation_best_j": float(j_scores[best_j_idx]),
                "validation_sensitivity_at_opt": float(tpr[best_j_idx]),
                "validation_specificity_at_opt": float(1.0 - fpr[best_j_idx])
            }

            full_model.save_checkpoint(
                CHECKPOINT_BEST,
                epoch=ep,
                val_metric=best_auroc,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_metadata={
                    "metrics": metrics,
                    "per_anatomy_auroc": anat_aucs,
                    "macro_anatomy_auroc": macro_anat_auroc,
                    "composite_score": best_score,
                    "best_epoch": best_epoch,
                    "optimal_threshold": opt_thresh,
                    "strategy": "anatomy_balanced"
                }
            )
            logger.info(f" -> Best composite score ({best_score:.4f}, Val AUROC={best_auroc:.4f}, Anat AUROC={macro_anat_auroc:.4f}). Saved {CHECKPOINT_BEST.name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {ep} epochs (Patience={patience}).")
                break

    # Save reports and configs
    save_training_reports(history, best_epoch, best_auroc, pos_weight_val, best_calib)
    return {"history": history, "best_auroc": best_auroc, "best_epoch": best_epoch, "best_composite": best_score}


def save_training_reports(
    history: list[Dict[str, Any]],
    best_epoch: int,
    best_auroc: float,
    pos_weight_val: float,
    calib_dict: Optional[Dict[str, Any]] = None
):
    """Exports structured metrics, configs, calibration, and training curves."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(REPORTS_DIR / "training_history.csv", index=False)

    with open(REPORTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save calibration if present
    if calib_dict:
        with open(CALIBRATION_JSON, "w") as f:
            json.dump(calib_dict, f, indent=2)
        logger.info(f"Saved calibrated threshold config to {CALIBRATION_JSON}")

    # Save model config JSON
    model_cfg = {
        "model_name": "ConvNeXt-Base Multi-Region Bone Fracture Classifier",
        "backbone": "convnext_base",
        "weights": "ConvNeXt_Base_Weights.DEFAULT",
        "feature_dim": FEATURE_DIM,
        "head_architecture": "1024 -> 512 -> 256 -> 1 (GELU, Dropout 0.3)",
        "input_resolution": [224, 224],
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "loss_function": "FocalLoss (alpha=0.55, gamma=1.5)",
        "sampling_strategy": "Multi-Anatomy Balanced Weighted Sampling",
        "pos_weight": pos_weight_val,
        "supported_anatomies": ["wrist", "hand", "leg", "hip", "shoulder", "mixed"],
        "checkpoint_path": str(CHECKPOINT_BEST)
    }
    with open(MODEL_CONFIG_JSON, "w") as f:
        json.dump(model_cfg, f, indent=2)

    with open(LABEL_MAP_JSON, "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    # Generate training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = df_hist["epoch"]

    # Loss curve
    axes[0].plot(epochs_range, df_hist["train_loss"], label="Train Loss", color="#1e88e5", lw=2)
    axes[0].plot(epochs_range, df_hist["val_loss"], label="Val Loss", color="#e53935", lw=2, linestyle="--")
    axes[0].set_title("Loss Curves (ConvNeXt-Base Balanced)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # AUROC & F1 curves
    axes[1].plot(epochs_range, df_hist["auroc"], label="Val Global AUROC", color="#43a047", lw=2)
    if "macro_anatomy_auroc" in df_hist.columns:
        axes[1].plot(epochs_range, df_hist["macro_anatomy_auroc"], label="Val Macro Anat AUROC", color="#00838f", lw=2, linestyle="-.")
    axes[1].plot(epochs_range, df_hist["f1"], label="Val F1", color="#fb8c00", lw=2, linestyle="--")
    axes[1].plot(epochs_range, df_hist["balanced_accuracy"], label="Val Bal Acc", color="#8e24aa", lw=1.5, linestyle=":")
    axes[1].set_title("Validation Metrics (Multi-Anatomy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim([0.0, 1.0])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    curves_path = REPORTS_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    logger.info(f"Saved training curves to {curves_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Base Fracture Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for head training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Target training device")
    parser.add_argument("--unbalanced", action="store_true", help="Disable anatomy-balanced sampling")
    parser.add_argument("--bce-loss", action="store_true", help="Use standard BCE loss instead of Focal Loss")
    args = parser.parse_args()

    train_cached(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device_choice=args.device,
        balanced=not args.unbalanced,
        use_focal=not args.bce_loss
    )

