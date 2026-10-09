"""
Evaluation Artifacts Generator for Bone Fracture Analysis
MedGuard AI - Bone Fracture Subsystem
Generates:
1. Standalone Precision-Recall Curve (pr_curve.png)
2. Per-Anatomy Comparison Bar Chart (per_anatomy_comparison.png)
3. Per-Dataset Comparison Bar Chart (per_source_comparison.png)
4. Comprehensive False-Negative Review Sheet (false_negative_review.csv)
5. Test Predictions Cache (test_predictions.csv)
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from torch.amp import autocast

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.tasks.fracture.config import (
    CHECKPOINT_BEST,
    MANIFEST_TEST,
    REPORTS_DIR,
    CALIBRATION_JSON,
    SUPPORTED_ANATOMIES
)
from ml.tasks.fracture.dataset import FractureDataset
from ml.tasks.fracture.model import ConvNeXtFractureClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_artifacts")


def generate_all_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating evaluation artifacts using device: {device}")

    # 1. Load threshold
    opt_thresh = 0.5200
    if CALIBRATION_JSON.exists():
        with open(CALIBRATION_JSON, "r") as f:
            c = json.load(f)
            opt_thresh = float(c.get("optimal_threshold_youden", 0.5200))
    logger.info(f"Decision threshold: {opt_thresh:.4f}")

    # 2. Run inference across untouched test manifest
    model = ConvNeXtFractureClassifier(pretrained=False).to(device)
    model.load_checkpoint(CHECKPOINT_BEST, device=device)
    model.eval()

    ds = FractureDataset(MANIFEST_TEST, is_training=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_probs = []
    all_targets = []
    all_sources = []
    all_anatomies = []
    all_paths = []
    all_ids = []

    logger.info(f"Extracting predictions for {len(ds)} test samples...")
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16 if device.type == "cuda" else torch.float32):
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().float().numpy())
            all_targets.append(batch["label"].numpy())
            all_sources.extend(batch["dataset_source"])
            all_anatomies.extend(batch["anatomical_region"])
            all_paths.extend(batch["image_path"])
            all_ids.extend(batch["image_id"])

    y_prob = np.vstack(all_probs).ravel()
    y_true = np.vstack(all_targets).ravel()
    y_pred = (y_prob >= opt_thresh).astype(int)

    # 3. Save full predictions dataframe
    df_preds = pd.DataFrame({
        "image_id": all_ids,
        "image_path": all_paths,
        "dataset_source": all_sources,
        "anatomical_region": all_anatomies,
        "true_label": y_true.astype(int),
        "predicted_probability": np.round(y_prob, 5),
        "predicted_label": y_pred,
        "decision_threshold": opt_thresh
    })
    preds_csv = REPORTS_DIR / "test_predictions.csv"
    df_preds.to_csv(preds_csv, index=False)
    logger.info(f"Saved test predictions to {preds_csv}")

    # 4. Generate False-Negative Review Sheet (Actual Fracture = 1, Predicted = 0)
    df_fn = df_preds[(df_preds["true_label"] == 1) & (df_preds["predicted_label"] == 0)].copy()
    df_fn["confidence_gap"] = np.round(opt_thresh - df_fn["predicted_probability"], 5)
    # Severity risk classification based on proximity to threshold
    df_fn["risk_tier"] = np.where(
        df_fn["confidence_gap"] < 0.10,
        "Borderline (Margin < 0.10)",
        np.where(df_fn["confidence_gap"] < 0.25, "Moderate Miss (Margin < 0.25)", "Severe False Negative")
    )
    df_fn = df_fn.sort_values(by="predicted_probability", ascending=False)
    fn_csv = REPORTS_DIR / "false_negative_review.csv"
    df_fn.to_csv(fn_csv, index=False)
    logger.info(f"Saved {len(df_fn)} False Negative cases to {fn_csv}")

    # 5. Standalone PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, color="#2e7d32", lw=2.5, label=f"ConvNeXt-Base (PR-AUC = {ap:.4f})")
    plt.axhline(y=float(y_true.mean()), color="#9e9e9e", linestyle="--", label=f"Baseline Prevalence ({y_true.mean():.3f})")
    plt.title("Precision-Recall Curve (Untouched Test Split - 3,448 Radiographs)", fontsize=12, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    pr_curve_path = REPORTS_DIR / "pr_curve.png"
    plt.savefig(pr_curve_path, dpi=200)
    plt.close()
    logger.info(f"Saved standalone PR curve to {pr_curve_path}")

    # 6. Per-Anatomy Performance Bar Chart
    df_anatomy = pd.read_csv(REPORTS_DIR / "anatomy_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    regions = df_anatomy["anatomical_region"].tolist()
    sens = df_anatomy["sensitivity"].tolist()
    spec = df_anatomy["specificity"].tolist()
    x = np.arange(len(regions))
    width = 0.35

    ax.bar(x - width/2, [s * 100 for s in sens], width, label="Sensitivity (%)", color="#1e88e5")
    ax.bar(x + width/2, [s * 100 for s in spec], width, label="Specificity (%)", color="#43a047")
    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_title("Test Set Performance by Anatomical Region (ConvNeXt-Base)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regions], fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    anatomy_plot_path = REPORTS_DIR / "per_anatomy_comparison.png"
    plt.savefig(anatomy_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved per-anatomy comparison to {anatomy_plot_path}")

    # 7. Per-Dataset Performance Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    with open(REPORTS_DIR / "test_metrics.json", "r") as f:
        metrics_meta = json.load(f)
    sources = metrics_meta["by_dataset_source"]
    s_names = ["Graz Wrist (15 GB)", "FracAtlas"]
    s_sens = [s["sensitivity"] * 100 for s in sources]
    s_spec = [s["specificity"] * 100 for s in sources]
    s_auroc = [s["auroc"] * 100 for s in sources]
    sx = np.arange(len(s_names))
    swidth = 0.25

    ax.bar(sx - swidth, s_sens, swidth, label="Sensitivity (%)", color="#1e88e5")
    ax.bar(sx, s_spec, swidth, label="Specificity (%)", color="#43a047")
    ax.bar(sx + swidth, s_auroc, swidth, label="AUROC (%)", color="#fb8c00")
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Generalization: Graz Wrist vs Multi-Region FracAtlas", fontsize=12, fontweight="bold")
    ax.set_xticks(sx)
    ax.set_xticklabels(s_names, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    source_plot_path = REPORTS_DIR / "per_source_comparison.png"
    plt.savefig(source_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved per-source comparison to {source_plot_path}")

    logger.info("All evaluation artifacts generated successfully.")


if __name__ == "__main__":
    generate_all_artifacts()
