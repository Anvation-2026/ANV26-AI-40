"""
Model Evaluation, Calibration, and Anatomical Breakdown Module
MedGuard AI - Bone Fracture Analysis Subsystem
Genuine ConvNeXt-Base Multi-Region Radiograph Evaluation

Evaluates the trained model on the untouched test split (3,448 samples).
Computes:
- Global AUROC, PR-AUC, Sensitivity, Specificity, F1, Balanced Accuracy, Brier score
- Per-anatomical region breakdown (wrist, leg, hand, mixed, hip, shoulder)
- Per-dataset-source breakdown (Graz vs FracAtlas)
- Optimal threshold calibration via Youden's J statistic
- ROC curves, PR curves, and Confusion Matrix plots
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.tasks.fracture.config import (
    MANIFEST_TEST,
    MANIFEST_VAL,
    CHECKPOINT_BEST,
    REPORTS_DIR,
    CALIBRATION_JSON,
    SUPPORTED_ANATOMIES
)
from ml.tasks.fracture.dataset import FractureDataset
from ml.tasks.fracture.model import ConvNeXtFractureClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fracture_evaluate")


def run_inference_on_manifest(
    model: ConvNeXtFractureClassifier,
    manifest_path: Path,
    device: torch.device,
    batch_size: int = 16
) -> Dict[str, Any]:
    """Runs batched inference across a manifest and collects predictions, targets, and metadata."""
    ds = FractureDataset(manifest_path, is_training=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    all_probs = []
    all_targets = []
    all_sources = []
    all_anatomies = []
    all_paths = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().float().numpy())
            all_targets.append(batch["label"].numpy())
            all_sources.extend(batch["dataset_source"])
            all_anatomies.extend(batch["anatomical_region"])
            all_paths.extend(batch["image_path"])

    probs_arr = np.vstack(all_probs).ravel()
    targets_arr = np.vstack(all_targets).ravel()

    return {
        "probs": probs_arr,
        "targets": targets_arr,
        "sources": all_sources,
        "anatomies": all_anatomies,
        "paths": all_paths
    }


def calibrate_threshold(val_targets: np.ndarray, val_probs: np.ndarray) -> Dict[str, float]:
    """
    Computes optimal clinical decision thresholds using Youden's J statistic
    strictly on the held-out validation set.
    """
    fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])
    
    # Also compute high-sensitivity screening threshold (Sensitivity >= 95%)
    sens_95_idx = np.where(tpr >= 0.95)[0]
    high_sens_threshold = float(thresholds[sens_95_idx[0]]) if len(sens_95_idx) > 0 else optimal_threshold

    calib = {
        "optimal_threshold_youden": round(optimal_threshold, 4),
        "high_sensitivity_threshold": round(high_sens_threshold, 4),
        "validation_best_j": float(j_scores[best_idx]),
        "validation_sensitivity_at_opt": float(tpr[best_idx]),
        "validation_specificity_at_opt": float(1.0 - fpr[best_idx])
    }
    
    CALIBRATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_JSON, "w") as f:
        json.dump(calib, f, indent=2)
    logger.info(f"Saved calibrated threshold config to {CALIBRATION_JSON}")
    return calib


def evaluate_fracture_model(checkpoint_path: Path = CHECKPOINT_BEST):
    """Full independent test evaluation pipeline."""
    logger.info("=== Starting Independent Fracture Model Evaluation ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model = ConvNeXtFractureClassifier(pretrained=False).to(device)
    model.load_checkpoint(checkpoint_path, device=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # 1. Calibrate threshold on validation set
    logger.info("Running validation inference for threshold calibration...")
    val_res = run_inference_on_manifest(model, MANIFEST_VAL, device)
    calib = calibrate_threshold(val_res["targets"], val_res["probs"])
    opt_thresh = calib["optimal_threshold_youden"]
    logger.info(f"Using Youden's J calibrated threshold: {opt_thresh:.4f}")

    # 2. Independent evaluation on untouched test set
    logger.info("Running inference on untouched held-out TEST split (3,448 samples)...")
    test_res = run_inference_on_manifest(model, MANIFEST_TEST, device)
    y_true = test_res["targets"]
    y_prob = test_res["probs"]
    y_pred = (y_prob >= opt_thresh).astype(int)

    # Global test metrics
    auroc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    global_metrics = {
        "test_samples": int(len(y_true)),
        "decision_threshold": opt_thresh,
        "auroc": auroc,
        "average_precision": ap,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "brier_score": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

    logger.info("\n=======================================================")
    logger.info("GLOBAL TEST SET METRICS (UNTOUCHED TEST SPLIT)")
    logger.info("=======================================================")
    logger.info(f"Test Samples: {len(y_true):,} | Calibrated Threshold: {opt_thresh:.4f}")
    logger.info(f"AUROC:        {auroc:.4f}")
    logger.info(f"AP / PR-AUC:  {ap:.4f}")
    logger.info(f"Sensitivity:  {sens:.4f} ({tp}/{tp+fn})")
    logger.info(f"Specificity:  {spec:.4f} ({tn}/{tn+fp})")
    logger.info(f"Precision:    {prec:.4f}")
    logger.info(f"F1-Score:     {f1:.4f}")
    logger.info(f"Balanced Acc: {bal_acc:.4f}")
    logger.info(f"Brier Score:  {brier:.4f}")
    logger.info(f"Confusion:    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    logger.info("=======================================================\n")

    # 3. Per-Anatomical Region Breakdown
    df_test_meta = pd.DataFrame({
        "target": y_true,
        "prob": y_prob,
        "pred": y_pred,
        "source": test_res["sources"],
        "anatomy": test_res["anatomies"]
    })

    anatomy_records = []
    for region in SUPPORTED_ANATOMIES:
        df_reg = df_test_meta[df_test_meta["anatomy"] == region]
        if len(df_reg) == 0:
            continue
        n_samples = len(df_reg)
        n_pos = int((df_reg["target"] == 1).sum())
        n_neg = int((df_reg["target"] == 0).sum())
        
        try:
            reg_auroc = float(roc_auc_score(df_reg["target"], df_reg["prob"])) if (n_pos > 0 and n_neg > 0) else float("nan")
        except Exception:
            reg_auroc = float("nan")

        reg_cm = confusion_matrix(df_reg["target"], df_reg["pred"], labels=[0, 1])
        r_tn, r_fp, r_fn, r_tp = reg_cm.ravel()
        r_sens = float(r_tp / (r_tp + r_fn)) if (r_tp + r_fn) > 0 else 0.0
        r_spec = float(r_tn / (r_tn + r_fp)) if (r_tn + r_fp) > 0 else 0.0
        r_f1 = float(f1_score(df_reg["target"], df_reg["pred"], zero_division=0))

        rec = {
            "anatomical_region": region,
            "total_samples": n_samples,
            "positives": n_pos,
            "negatives": n_neg,
            "sensitivity": round(r_sens, 4),
            "specificity": round(r_spec, 4),
            "f1_score": round(r_f1, 4),
            "auroc": round(reg_auroc, 4) if not np.isnan(reg_auroc) else "N/A"
        }
        anatomy_records.append(rec)

    df_anatomy = pd.DataFrame(anatomy_records)
    anatomy_csv_path = REPORTS_DIR / "anatomy_metrics.csv"
    df_anatomy.to_csv(anatomy_csv_path, index=False)
    logger.info(f"Exported per-anatomy metrics to {anatomy_csv_path}")

    # 4. Per-Dataset Source Breakdown
    source_records = []
    for src in df_test_meta["source"].unique():
        df_src = df_test_meta[df_test_meta["source"] == src]
        src_pos = int((df_src["target"] == 1).sum())
        src_neg = int((df_src["target"] == 0).sum())
        src_auroc = float(roc_auc_score(df_src["target"], df_src["prob"])) if (src_pos > 0 and src_neg > 0) else float("nan")
        src_cm = confusion_matrix(df_src["target"], df_src["pred"], labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = src_cm.ravel()
        source_records.append({
            "source": src,
            "samples": len(df_src),
            "positives": src_pos,
            "negatives": src_neg,
            "auroc": round(src_auroc, 4),
            "sensitivity": round(float(s_tp / (s_tp + s_fn)), 4) if (s_tp + s_fn) > 0 else 0.0,
            "specificity": round(float(s_tn / (s_tn + s_fp)), 4) if (s_tn + s_fp) > 0 else 0.0
        })

    # Save complete test metrics JSON
    final_report = {
        "global_test_metrics": global_metrics,
        "calibration": calib,
        "by_anatomical_region": anatomy_records,
        "by_dataset_source": source_records
    }
    with open(REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(final_report, f, indent=2)

    # 5. Generate Visual Diagnostic Plots
    plot_evaluation_figures(y_true, y_prob, y_pred, auroc, ap)

    logger.info("=== Independent Test Evaluation Complete ===")
    return final_report


def plot_evaluation_figures(y_true, y_prob, y_pred, auroc, ap):
    """Generates ROC curve, PR curve, and Confusion Matrix figures."""
    # 1. ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, color="#1e88e5", lw=2, label=f"ConvNeXt-Base (AUROC = {auroc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", linestyle="--")
    axes[0].set_title("Receiver Operating Characteristic (ROC) - Test Split")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, color="#43a047", lw=2, label=f"ConvNeXt-Base (PR-AUC = {ap:.4f})")
    axes[1].set_title("Precision-Recall Curve - Test Split")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # 2. Confusion Matrix (pure matplotlib)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix - Test Split")
    fig.colorbar(im)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(["No Fracture", "Fracture"])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(["No Fracture", "Fracture"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    evaluate_fracture_model()
