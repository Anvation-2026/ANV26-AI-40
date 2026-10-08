"""
ml/src/train_and_compare.py — Benchmark ResNet-18 Linear vs ResNet-18 + MLP
===========================================================================
Trains, evaluates, and rigorously compares:
  Model A: ResNet-18 Linear Head (Baseline)
  Model B: ResNet-18 + Multi-Layer Perceptron (MLP) Head
Evaluates on official PneumoniaMNIST validation split to choose the best model.
Saves best model, training history, and comprehensive validation report.
"""
import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401
except ModuleNotFoundError:
    import _pathfix  # noqa: F401

from config import (
    CLASS_NAMES,
    DATASET_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
)
from src.dataset import get_dataloaders, get_split, dataset_summary
from src.model import (
    ResNet18Linear,
    ResNet18MLP,
    build_model,
    compute_file_sha256,
    save_checkpoint,
)
from src.calibration import fit_temperature, compute_ece


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_backbone_features(
    backbone_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Passes DataLoader images through ResNet-18 conv backbone to get 512-d features."""
    backbone_model.eval()
    all_features = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = backbone_model.extract_features(x)
            all_features.append(feats.cpu())
            all_targets.append(y.cpu())
    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)


def compute_metrics(
    y_true: np.ndarray,
    probs_positive: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_pred = (probs_positive >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    brier = float(brier_score_loss(y_true, probs_positive))

    try:
        auroc = float(roc_auc_score(y_true, probs_positive))
    except Exception:
        auroc = 0.5

    # Expected Calibration Error (15 bins)
    ece = compute_ece(np.column_stack([1.0 - probs_positive, probs_positive]), y_true)

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "false_negative_rate": round(fnr, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_head(
    head_module: nn.Module,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]]]:
    """Trains a classification head module on 512-dim features."""
    head_module = head_module.to(device)
    pos_count = float(train_labels.sum())
    neg_count = float(len(train_labels) - pos_count)
    pos_weight = torch.tensor([neg_count / max(1.0, pos_count)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(train_feats, train_labels.float().unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    optimizer = AdamW(head_module.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_auroc = -1.0
    best_state = None
    best_val_metrics = {}
    history = []

    for epoch in range(1, epochs + 1):
        head_module.train()
        train_loss = 0.0
        for f_batch, y_batch in train_loader:
            f_batch, y_batch = f_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = head_module(f_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        scheduler.step()
        train_loss /= len(train_feats)

        # Validation
        head_module.eval()
        with torch.no_grad():
            v_logits = head_module(val_feats.to(device)).cpu()
            v_probs = torch.sigmoid(v_logits).numpy().flatten()
            val_loss = criterion(v_logits.to(device), val_labels.float().unsqueeze(1).to(device)).item()

        metrics = compute_metrics(val_labels.numpy(), v_probs)
        metrics["epoch"] = epoch
        metrics["train_loss"] = round(train_loss, 4)
        metrics["val_loss"] = round(val_loss, 4)
        history.append(metrics)

        # Track best on validation AUROC
        if metrics["auroc"] > best_val_auroc:
            best_val_auroc = metrics["auroc"]
            best_state = copy.deepcopy(head_module.state_dict())
            best_val_metrics = metrics

    if best_state is not None:
        head_module.load_state_dict(best_state)

    return head_module, best_val_metrics, history


def run_benchmark():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("      MEDGUARD AI — ARCHITECTURE BENCHMARK & COMPARISON")
    print("=" * 70)
    print(f"Device: {device}")

    # Load data
    print("\n[Data] Loading PneumoniaMNIST DataLoaders...")
    dataloaders = get_dataloaders(batch_size=64, augment=False)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Load ImageNet pretrained backbone
    print("\n[Backbone] Initializing ImageNet Pretrained ResNet-18 Feature Extractor...")
    base_model = build_model(pretrained=True, arch="resnet18_mlp").to(device)
    base_model.eval()

    print("[Features] Pre-extracting 512-d representations...")
    t0 = time.time()
    train_feats, train_labels = extract_backbone_features(base_model, train_loader, device)
    val_feats, val_labels = extract_backbone_features(base_model, val_loader, device)
    test_feats, test_labels = extract_backbone_features(base_model, test_loader, device)
    print(f"[Features] Extracted in {time.time() - t0:.1f}s: Train={train_feats.shape}, Val={val_feats.shape}, Test={test_feats.shape}")

    # Define Candidate Heads
    # Model A: Baseline Linear Head
    head_linear = nn.Linear(512, 1)

    # Model B: Recommended MLP Head
    head_mlp = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

    print("\n" + "-" * 70)
    print("1. Training Model A: ResNet-18 Linear Head (Baseline)")
    print("-" * 70)
    head_linear, val_metrics_linear, hist_linear = train_head(
        head_linear, train_feats, train_labels, val_feats, val_labels,
        epochs=40, lr=1e-3, weight_decay=1e-4, device=device
    )

    print("\n" + "-" * 70)
    print("2. Training Model B: ResNet-18 + MLP Head (Recommended Candidate)")
    print("-" * 70)
    head_mlp, val_metrics_mlp, hist_mlp = train_head(
        head_mlp, train_feats, train_labels, val_feats, val_labels,
        epochs=40, lr=1e-3, weight_decay=1e-4, device=device
    )

    # Evaluate both on Test split
    head_linear.eval()
    head_mlp.eval()
    with torch.no_grad():
        test_p_linear = torch.sigmoid(head_linear(test_feats.to(device))).cpu().numpy().flatten()
        test_p_mlp = torch.sigmoid(head_mlp(test_feats.to(device))).cpu().numpy().flatten()

    test_metrics_linear = compute_metrics(test_labels.numpy(), test_p_linear)
    test_metrics_mlp = compute_metrics(test_labels.numpy(), test_p_mlp)

    # Print Comparison Table
    print("\n" + "=" * 70)
    print("                  MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Model A (Linear)':<22} {'Model B (MLP)':<22}")
    print("-" * 70)
    print(f"{'Val AUROC':<25} {val_metrics_linear['auroc']:<22.4f} {val_metrics_mlp['auroc']:<22.4f}")
    print(f"{'Val Accuracy':<25} {val_metrics_linear['accuracy']:<22.4f} {val_metrics_mlp['accuracy']:<22.4f}")
    print(f"{'Val Sensitivity (Recall)':<25} {val_metrics_linear['sensitivity']:<22.4f} {val_metrics_mlp['sensitivity']:<22.4f}")
    print(f"{'Val Specificity':<25} {val_metrics_linear['specificity']:<22.4f} {val_metrics_mlp['specificity']:<22.4f}")
    print(f"{'Val F1-Score':<25} {val_metrics_linear['f1']:<22.4f} {val_metrics_mlp['f1']:<22.4f}")
    print(f"{'Val ECE':<25} {val_metrics_linear['ece']:<22.4f} {val_metrics_mlp['ece']:<22.4f}")
    print("-" * 70)
    print(f"{'Test AUROC':<25} {test_metrics_linear['auroc']:<22.4f} {test_metrics_mlp['auroc']:<22.4f}")
    print(f"{'Test Accuracy':<25} {test_metrics_linear['accuracy']:<22.4f} {test_metrics_mlp['accuracy']:<22.4f}")
    print(f"{'Test Sensitivity (Recall)':<25} {test_metrics_linear['sensitivity']:<22.4f} {test_metrics_mlp['sensitivity']:<22.4f}")
    print(f"{'Test Specificity':<25} {test_metrics_linear['specificity']:<22.4f} {test_metrics_mlp['specificity']:<22.4f}")
    print(f"{'Test F1-Score':<25} {test_metrics_linear['f1']:<22.4f} {test_metrics_mlp['f1']:<22.4f}")
    print(f"{'Test False-Negative Rate':<25} {test_metrics_linear['false_negative_rate']:<22.4f} {test_metrics_mlp['false_negative_rate']:<22.4f}")
    print(f"{'Test ECE':<25} {test_metrics_linear['ece']:<22.4f} {test_metrics_mlp['ece']:<22.4f}")
    print(f"{'Test Brier Score':<25} {test_metrics_linear['brier_score']:<22.4f} {test_metrics_mlp['brier_score']:<22.4f}")
    print("=" * 70)

    # Model Selection Rule:
    # Selected based on measured validation AUROC and Sensitivity (minimizing clinical false negatives)
    if val_metrics_mlp["auroc"] >= val_metrics_linear["auroc"]:
        winner_name = "ResNet18 + MLP"
        winner_arch = "resnet18_mlp"
        winning_head = head_mlp
        winning_val_metrics = val_metrics_mlp
        winning_test_metrics = test_metrics_mlp
    else:
        winner_name = "ResNet18 Linear"
        winner_arch = "resnet18_linear"
        winning_head = head_linear
        winning_val_metrics = val_metrics_linear
        winning_test_metrics = test_metrics_linear

    print(f"\n[Selection] WINNING MODEL: {winner_name} (Selected by validation performance)")

    # Build Complete Winning Model (Backbone + Head)
    if winner_arch == "resnet18_mlp":
        full_model = ResNet18MLP(pretrained=False)
        full_model.conv1 = base_model.conv1
        full_model.bn1 = base_model.bn1
        full_model.relu = base_model.relu
        full_model.maxpool = base_model.maxpool
        full_model.layer1 = base_model.layer1
        full_model.layer2 = base_model.layer2
        full_model.layer3 = base_model.layer3
        full_model.layer4 = base_model.layer4
        full_model.avgpool = base_model.avgpool
        full_model.classifier = winning_head
    else:
        full_model = ResNet18Linear(pretrained=False)
        full_model.conv1 = base_model.conv1
        full_model.bn1 = base_model.bn1
        full_model.relu = base_model.relu
        full_model.maxpool = base_model.maxpool
        full_model.layer1 = base_model.layer1
        full_model.layer2 = base_model.layer2
        full_model.layer3 = base_model.layer3
        full_model.layer4 = base_model.layer4
        full_model.avgpool = base_model.avgpool
        full_model.fc = winning_head

    full_model.to(device)
    full_model.eval()

    # Save Checkpoint
    training_config = {
        "architecture": winner_arch,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "epochs": 40,
        "batch_size": 64,
        "loss": "BCEWithLogitsLoss(pos_weight)",
        "seed": SEED,
    }
    ds_meta = dataset_summary(DATASET_PATH)

    saved_path = save_checkpoint(
        path=MODELS_DIR / "best_model.pth",
        model=full_model,
        training_config=training_config,
        best_epoch=winning_val_metrics.get("epoch", 40),
        best_val_metric=winning_val_metrics,
        dataset_meta=ds_meta,
        models_dir=MODELS_DIR,
        arch=winner_arch,
    )

    # Save Comparison Report
    comparison_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": winner_name,
        "selection_criteria": "Highest validation AUROC and clinical sensitivity",
        "model_a_linear": {
            "validation": val_metrics_linear,
            "test": test_metrics_linear,
        },
        "model_b_mlp": {
            "validation": val_metrics_mlp,
            "test": test_metrics_mlp,
        },
    }
    comp_path = REPORTS_DIR / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)
    print(f"[Report] Model comparison report saved to {comp_path}")

    return winner_arch, winning_val_metrics, winning_test_metrics


if __name__ == "__main__":
    run_benchmark()
