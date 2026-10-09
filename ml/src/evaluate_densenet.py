"""
ml/src/evaluate_densenet.py -- Comprehensive Evaluation & Calibration Pipeline for DenseNet-201.
================================================================================================
Computes:
  - Per-class and Macro AUROC
  - Per-class and Macro Average Precision (AP / PR-AUC)
  - Optimal per-class decision thresholds (tau_c) tuned on held-out validation data (Youden's J / F1)
  - Sensitivity, Specificity, Precision, F1, False Negatives, False Positives at tau_c
  - Expected Calibration Error (ECE) and Brier Score
  - Multi-label uncertainty estimation and abstention boundaries
  - Saves threshold artifacts to ml/models/densenet201/thresholds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Path bootstrap
try:
    from config import MODELS_DIR, REPORTS_DIR
except ModuleNotFoundError:
    from ..config import MODELS_DIR, REPORTS_DIR

try:
    from src.densenet_model import (
        CHESTMNIST_14_LABELS,
        NUM_CHEST_CLASSES,
        DenseNet201Chest14,
        load_densenet_checkpoint,
    )
    from src.chestmnist_dataset import get_chestmnist_dataloaders
except ModuleNotFoundError:
    from densenet_model import (
        CHESTMNIST_14_LABELS,
        NUM_CHEST_CLASSES,
        DenseNet201Chest14,
        load_densenet_checkpoint,
    )
    from chestmnist_dataset import get_chestmnist_dataloaders


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for binary predictions in [0, 1].
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)
    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(targets[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def find_optimal_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    method: str = "youden",
) -> Tuple[float, Dict[str, float]]:
    """
    Finds optimal decision threshold on validation set.
    Methods:
      - 'youden': Maximizes Sensitivity + Specificity - 1
      - 'f1': Maximizes F1 score
    """
    best_thresh = 0.50
    best_score = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (targets == 1)))
        tn = int(np.sum((preds == 0) & (targets == 0)))
        fp = int(np.sum((preds == 1) & (targets == 0)))
        fn = int(np.sum((preds == 0) & (targets == 1)))

        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        prec = tp / max(1, tp + fp)
        f1 = (2 * prec * sens) / max(1e-6, prec + sens)
        youden = sens + spec - 1.0

        score = youden if method == "youden" else f1
        if score > best_score:
            best_score = score
            best_thresh = float(t)
            best_metrics = {
                "threshold": round(float(t), 4),
                "sensitivity": round(float(sens), 4),
                "specificity": round(float(spec), 4),
                "precision": round(float(prec), 4),
                "f1": round(float(f1), 4),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }

    return best_thresh, best_metrics


def evaluate_predictions(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    tune_thresholds: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates multi-label predictions across all 14 diseases.
    """
    n_samples, n_classes = probs.shape
    per_class_results: Dict[str, Dict[str, Any]] = {}
    optimal_thresholds: Dict[str, float] = {}

    auroc_list = []
    ap_list = []
    f1_list = []
    sens_list = []
    spec_list = []
    ece_list = []

    for c in range(n_classes):
        label_name = CHESTMNIST_14_LABELS[c]
        c_probs = probs[:, c]
        c_targets = targets[:, c]

        # AUROC & AP
        n_pos = int(np.sum(c_targets == 1))
        n_neg = int(np.sum(c_targets == 0))

        if n_pos > 0 and n_neg > 0:
            try:
                c_auroc = float(roc_auc_score(c_targets, c_probs))
            except Exception:
                c_auroc = 0.50
            try:
                c_ap = float(average_precision_score(c_targets, c_probs))
            except Exception:
                c_ap = float(n_pos / max(1, n_samples))
        else:
            c_auroc = 0.50
            c_ap = 0.0

        auroc_list.append(c_auroc)
        ap_list.append(c_ap)

        # Threshold selection
        if tune_thresholds or thresholds is None or label_name not in thresholds:
            t_opt, t_metrics = find_optimal_threshold(c_probs, c_targets, method="youden")
            optimal_thresholds[label_name] = t_opt
        else:
            t_opt = thresholds[label_name]
            optimal_thresholds[label_name] = t_opt
            preds = (c_probs >= t_opt).astype(int)
            tp = int(np.sum((preds == 1) & (c_targets == 1)))
            tn = int(np.sum((preds == 0) & (c_targets == 0)))
            fp = int(np.sum((preds == 1) & (c_targets == 0)))
            fn = int(np.sum((preds == 0) & (c_targets == 1)))
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            prec = tp / max(1, tp + fp)
            f1 = (2 * prec * sens) / max(1e-6, prec + sens)
            t_metrics = {
                "threshold": round(float(t_opt), 4),
                "sensitivity": round(float(sens), 4),
                "specificity": round(float(spec), 4),
                "precision": round(float(prec), 4),
                "f1": round(float(f1), 4),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }

        # Calibration
        c_ece = compute_ece(c_probs, c_targets, n_bins=10)
        c_brier = float(np.mean((c_probs - c_targets) ** 2))
        ece_list.append(c_ece)

        f1_list.append(t_metrics["f1"])
        sens_list.append(t_metrics["sensitivity"])
        spec_list.append(t_metrics["specificity"])

        per_class_results[label_name] = {
            "auroc": round(c_auroc, 4),
            "average_precision": round(c_ap, 4),
            "prevalence_percent": round(float(n_pos / max(1, n_samples)) * 100.0, 2),
            "positives_count": n_pos,
            "negatives_count": n_neg,
            "ece": round(c_ece, 4),
            "brier_score": round(c_brier, 4),
            **t_metrics,
        }

    macro_summary = {
        "macro_auroc": round(float(np.mean(auroc_list)), 4),
        "macro_average_precision": round(float(np.mean(ap_list)), 4),
        "macro_ap": round(float(np.mean(ap_list)), 4),
        "macro_sensitivity": round(float(np.mean(sens_list)), 4),
        "macro_specificity": round(float(np.mean(spec_list)), 4),
        "macro_f1": round(float(np.mean(f1_list)), 4),
        "macro_ece": round(float(np.mean(ece_list)), 4),
        "total_samples": n_samples,
    }

    return {
        "macro": macro_summary,
        "per_class": per_class_results,
        "optimal_thresholds": optimal_thresholds,
    }


def evaluate_densenet_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs forward pass through entire DataLoader and returns all probabilities and targets.
    """
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def save_thresholds_file(
    thresholds: Dict[str, float],
    metrics: Dict[str, Any],
    out_path: Path = MODELS_DIR / "densenet201" / "thresholds.json",
) -> None:
    """
    Saves calibrated per-class decision thresholds to JSON.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "densenet201_mlp_1920_512_256_14",
        "dataset": "ChestMNIST 224x224 (NIH-ChestXray14)",
        "tuning_split": "validation",
        "method": "youden_j_maximization",
        "thresholds": thresholds,
        "abstention_margin": 0.10,
        "macro_val_auroc": metrics.get("macro", {}).get("macro_auroc"),
        "macro_val_ap": metrics.get("macro", {}).get("macro_average_precision"),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Thresholds] Saved per-class thresholds to: {out_path}")
