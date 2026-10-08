import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Path bootstrap
from src import _ML_ROOT  # noqa: F401

from config import (
    CLASS_NAMES,
    DATASET_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    STANDARD_LIMITATIONS,
)
from src.dataset import dataset_summary, get_dataloaders
from src.model import load_checkpoint


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs forward pass over a DataLoader.
    Returns:
        logits: (N, 2)
        probs: (N, 2)
        targets: (N,)
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    logits_arr = torch.cat(all_logits, dim=0).numpy()
    targets_arr = torch.cat(all_targets, dim=0).numpy().astype(int)
    # Standard softmax probabilities
    exp_l = np.exp(logits_arr - np.max(logits_arr, axis=1, keepdims=True))
    probs_arr = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    return logits_arr, probs_arr, targets_arr


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob_positive: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes all core binary classification metrics."""
    y_pred = (y_prob_positive >= threshold).astype(int)

    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auroc = float(roc_auc_score(y_true, y_prob_positive))
    except Exception:
        auroc = 0.5

    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_labels": [["TN", "FP"], ["FN", "TP"]],
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "sensitivity": round(sensitivity, 4),
        "recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total},
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Computes ECE (Expected Calibration Error), Brier score, and NLL."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    p_pos = probs[:, 1]
    brier = float(brier_score_loss(y_true, p_pos))
    nll = float(log_loss(y_true, probs))

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(brier, 4),
        "nll": round(nll, 4),
    }


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_prob_positive: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = SEED,
) -> Dict[str, Dict[str, float]]:
    """Computes 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_accs: List[float] = []
    boot_sens: List[float] = []
    boot_specs: List[float] = []
    boot_aurocs: List[float] = []

    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idxs]
        yp_sample = y_prob_positive[idxs]
        preds = (yp_sample >= 0.5).astype(int)

        tp = np.sum((preds == 1) & (yt_sample == 1))
        fn = np.sum((preds == 0) & (yt_sample == 1))
        tn = np.sum((preds == 0) & (yt_sample == 0))
        fp = np.sum((preds == 1) & (yt_sample == 0))

        boot_accs.append(np.mean(preds == yt_sample))
        boot_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        boot_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        try:
            boot_aurocs.append(roc_auc_score(yt_sample, yp_sample))
        except Exception:
            pass

    def get_ci(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"ci_lower": 0.0, "ci_upper": 0.0}
        return {
            "ci_lower": round(float(np.percentile(values, 2.5)), 4),
            "ci_upper": round(float(np.percentile(values, 97.5)), 4),
        }

    return {
        "accuracy": get_ci(boot_accs),
        "sensitivity": get_ci(boot_sens),
        "specificity": get_ci(boot_specs),
        "auroc": get_ci(boot_aurocs),
    }


def evaluate_basic(
    model_path: Path = MODELS_DIR / "best_model.pth",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Phase 1 basic test metrics evaluator."""
    device = torch.device(device_str)
    model, ckpt_meta = load_checkpoint(model_path, device=device_str)

    dataloaders = get_dataloaders(batch_size=64, augment=False, path=DATASET_PATH)
    _, probs, targets = collect_predictions(model, dataloaders["test"], device)

    p_pneumonia = probs[:, 1]
    metrics = compute_binary_metrics(targets, p_pneumonia)

    result = {
        "model_version": ckpt_meta.get("full_model_version", "unknown"),
        "checkpoint_sha256": ckpt_meta.get("sha256", "unknown"),
        "evaluated_split": "test",
        "num_samples": len(targets),
        "metrics": metrics,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "test_metrics_basic.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 55)
    print("      PHASE 1 BASIC TEST METRICS EVALUATION")
    print("=" * 55)
    print(f"Model: {result['model_version']}")
    print(f"Test Samples: {len(targets)}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity:         {metrics['specificity']:.4f}")
    print(f"F1 Score:            {metrics['f1']:.4f}")
    print(f"AUROC:               {metrics['auroc']:.4f}")
    print(f"False-Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print("=" * 55)
    print(f"Saved to {out_file}\n")

    return result


def evaluate_full(
    model_path: Path = MODELS_DIR / "best_model.pth",
    models_dir: Path = MODELS_DIR,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Full Phase 2+ validation report generation for both val and test splits.
    Includes bootstrap CIs, calibration before/after, selective prediction,
    and merges quality/OOD eval if present.
    """
    device = torch.device(device_str)
    model, ckpt_meta = load_checkpoint(model_path, device=device_str)
    ds_meta = dataset_summary(DATASET_PATH)

    dataloaders = get_dataloaders(batch_size=64, augment=False, path=DATASET_PATH)

    # Predictions on val and test
    val_logits, val_probs_raw, val_targets = collect_predictions(model, dataloaders["val"], device)
    test_logits, test_probs_raw, test_targets = collect_predictions(model, dataloaders["test"], device)

    # Load temperature calibration if available
    calib_file = models_dir / "calibration.json"
    temperature = 1.0
    calib_meta = None
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_meta = json.load(f)
            temperature = float(calib_meta.get("temperature", 1.0))

    # Apply temperature
    def scale_probs(logits: np.ndarray, temp: float) -> np.ndarray:
        scaled_logits = logits / max(1e-4, temp)
        exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    val_probs_calib = scale_probs(val_logits, temperature)
    test_probs_calib = scale_probs(test_logits, temperature)

    # Raw metrics
    val_raw_metrics = compute_binary_metrics(val_targets, val_probs_raw[:, 1])
    test_raw_metrics = compute_binary_metrics(test_targets, test_probs_raw[:, 1])

    # Calibrated metrics
    val_calib_metrics = compute_binary_metrics(val_targets, val_probs_calib[:, 1])
    test_calib_metrics = compute_binary_metrics(test_targets, test_probs_calib[:, 1])

    # Calibration errors
    val_calib_err_raw = compute_calibration_metrics(val_targets, val_probs_raw)
    val_calib_err_cal = compute_calibration_metrics(val_targets, val_probs_calib)
    test_calib_err_raw = compute_calibration_metrics(test_targets, test_probs_raw)
    test_calib_err_cal = compute_calibration_metrics(test_targets, test_probs_calib)

    # Bootstrap CIs (on test split, calibrated)
    test_boot_cis = compute_bootstrap_cis(test_targets, test_probs_calib[:, 1])

    # Load thresholds for selective prediction
    thresholds_file = models_dir / "thresholds.json"
    tau_accept = 0.80
    thresh_meta = {}
    if thresholds_file.exists():
        with open(thresholds_file, "r", encoding="utf-8") as f:
            thresh_meta = json.load(f)
            tau_accept = float(thresh_meta.get("tau_accept", 0.80))

    # Selective prediction analysis on test split
    test_confs = np.max(test_probs_calib, axis=1)
    accepted_mask = test_confs >= tau_accept
    abstained_mask = ~accepted_mask

    accepted_count = int(np.sum(accepted_mask))
    abstained_count = int(np.sum(abstained_mask))
    total_test = len(test_targets)
    coverage = float(accepted_count / total_test) if total_test > 0 else 0.0

    accepted_targets = test_targets[accepted_mask]
    accepted_probs = test_probs_calib[accepted_mask, 1]
    abstained_targets = test_targets[abstained_mask]
    abstained_probs = test_probs_calib[abstained_mask, 1]

    if accepted_count > 0:
        sel_acc_metrics = compute_binary_metrics(accepted_targets, accepted_probs)
        accepted_error_rate = 1.0 - sel_acc_metrics["accuracy"]
    else:
        sel_acc_metrics = {"accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}
        accepted_error_rate = 0.0

    if abstained_count > 0:
        sel_abs_metrics = compute_binary_metrics(abstained_targets, abstained_probs)
        abstained_error_rate = 1.0 - sel_abs_metrics["accuracy"]
    else:
        sel_abs_metrics = {"accuracy": 0.0}
        abstained_error_rate = 0.0

    selective_eval = {
        "tau_accept": tau_accept,
        "coverage": round(coverage, 4),
        "total_test_samples": total_test,
        "accepted_count": accepted_count,
        "abstained_count": abstained_count,
        "accepted_accuracy": round(sel_acc_metrics["accuracy"], 4),
        "accepted_sensitivity": round(sel_acc_metrics["sensitivity"], 4),
        "accepted_specificity": round(sel_acc_metrics["specificity"], 4),
        "accepted_error_rate": round(accepted_error_rate, 4),
        "abstained_error_rate": round(abstained_error_rate, 4),
    }

    # Error analysis: False Negatives and False Positives on test
    test_preds_calib = (test_probs_calib[:, 1] >= 0.5).astype(int)
    fn_indices = np.where((test_preds_calib == 0) & (test_targets == 1))[0].tolist()
    fp_indices = np.where((test_preds_calib == 1) & (test_targets == 0))[0].tolist()

    fn_abstained = int(np.sum(abstained_mask[fn_indices]))
    fp_abstained = int(np.sum(abstained_mask[fp_indices]))

    error_analysis = {
        "test_total_fn": len(fn_indices),
        "test_fn_abstained": fn_abstained,
        "test_fn_accepted": len(fn_indices) - fn_abstained,
        "test_total_fp": len(fp_indices),
        "test_fp_abstained": fp_abstained,
        "test_fp_accepted": len(fp_indices) - fp_abstained,
        "fn_sample_indices": fn_indices[:12],
    }

    # Load optional quality & OOD eval reports if they exist
    quality_eval = None
    quality_eval_file = REPORTS_DIR / "quality_eval.json"
    if quality_eval_file.exists():
        with open(quality_eval_file, "r", encoding="utf-8") as f:
            quality_eval = json.load(f)

    ood_eval = None
    ood_eval_file = REPORTS_DIR / "ood_eval.json"
    if ood_eval_file.exists():
        with open(ood_eval_file, "r", encoding="utf-8") as f:
            ood_eval = json.load(f)

    report_json = {
        "provenance": {
            "model_version": ckpt_meta.get("full_model_version", "unknown"),
            "checkpoint_sha256": ckpt_meta.get("sha256", "unknown"),
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "device": str(device),
            "split_sizes": ds_meta["split_sizes"],
        },
        "dataset": {
            "name": ds_meta["dataset_name"],
            "class_counts": ds_meta["class_counts"],
            "exact_duplicates": ds_meta["exact_duplicates"],
            "image_size": [224, 224],
        },
        "metrics": {
            "test": {
                "uncalibrated": {**test_raw_metrics, **test_calib_err_raw},
                "calibrated": {**test_calib_metrics, **test_calib_err_cal, "bootstrap_95ci": test_boot_cis},
            },
            "val": {
                "uncalibrated": {**val_raw_metrics, **val_calib_err_raw},
                "calibrated": {**val_calib_metrics, **val_calib_err_cal},
            },
        },
        "calibration": {
            "temperature": round(temperature, 4),
            "method": "Temperature Scaling (Validation NLL Minimization)",
            "val_ece_before": val_calib_err_raw["ece"],
            "val_ece_after": val_calib_err_cal["ece"],
            "test_ece_before": test_calib_err_raw["ece"],
            "test_ece_after": test_calib_err_cal["ece"],
        },
        "selective": selective_eval,
        "error_analysis": error_analysis,
        "quality_eval": quality_eval,
        "ood_eval": ood_eval,
        "limitations": STANDARD_LIMITATIONS,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_json_path = REPORTS_DIR / "validation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    # Generate Markdown validation report
    report_md_path = REPORTS_DIR / "validation_report.md"
    generate_markdown_report(report_json, report_md_path)

    print(f"\n[Evaluate] Full validation report written to {report_json_path} and {report_md_path}")
    return report_json


def generate_markdown_report(report: Dict, out_path: Path) -> None:
    """Renders comprehensive validation report to clean GitHub-style Markdown."""
    prov = report["provenance"]
    t_cal = report["metrics"]["test"]["calibrated"]
    v_cal = report["metrics"]["val"]["calibrated"]
    sel = report["selective"]
    cal = report["calibration"]
    err = report["error_analysis"]

    ci = t_cal.get("bootstrap_95ci", {})

    md = f"""# MedGuard AI — Model Validation Report

**Model Version:** `{prov['model_version']}`  
**Evaluated UTC:** `{prov['evaluated_utc']}`  
**Hardware Device:** `{prov['device']}`  
**Dataset:** PneumoniaMNIST+ 224x224 (Pediatric Chest X-Ray Benchmark)

---

## 1. Executive Summary & Test Performance

Primary evaluation on official **Test split** (N = {prov['split_sizes']['test']}):

| Metric | Test Value (Calibrated) | 95% Bootstrap CI | Validation Value |
|---|---|---|---|
| **AUROC** | **{t_cal['auroc']:.4f}** | [{ci.get('auroc', {}).get('ci_lower', 0):.4f}, {ci.get('auroc', {}).get('ci_upper', 0):.4f}] | {v_cal['auroc']:.4f} |
| **Accuracy** | **{t_cal['accuracy']:.4f}** | [{ci.get('accuracy', {}).get('ci_lower', 0):.4f}, {ci.get('accuracy', {}).get('ci_upper', 0):.4f}] | {v_cal['accuracy']:.4f} |
| **Sensitivity (Recall)** | **{t_cal['sensitivity']:.4f}** | [{ci.get('sensitivity', {}).get('ci_lower', 0):.4f}, {ci.get('sensitivity', {}).get('ci_upper', 0):.4f}] | {v_cal['sensitivity']:.4f} |
| **Specificity** | **{t_cal['specificity']:.4f}** | [{ci.get('specificity', {}).get('ci_lower', 0):.4f}, {ci.get('specificity', {}).get('ci_upper', 0):.4f}] | {v_cal['specificity']:.4f} |
| **Precision** | **{t_cal['precision']:.4f}** | - | {v_cal['precision']:.4f} |
| **F1 Score** | **{t_cal['f1']:.4f}** | - | {v_cal['f1']:.4f} |
| **False-Negative Rate** | **{t_cal['false_negative_rate']:.4f}** | - | {v_cal['false_negative_rate']:.4f} |

### Confusion Matrix (Test Split)
```
                Predicted Normal    Predicted Pneumonia
Actual Normal        {t_cal['confusion_matrix'][0][0]:<19} {t_cal['confusion_matrix'][0][1]}
Actual Pneumonia     {t_cal['confusion_matrix'][1][0]:<19} {t_cal['confusion_matrix'][1][1]}
```

---

## 2. Calibration & Temperature Scaling

Fitted on **Validation Split** (N = {prov['split_sizes']['val']}):
- **Learned Temperature $T$:** `{cal['temperature']:.4f}`
- **Validation ECE:** `{cal['val_ece_before']:.4f}` $\\to$ `{cal['val_ece_after']:.4f}`
- **Test ECE:** `{cal['test_ece_before']:.4f}` $\\to$ `{cal['test_ece_after']:.4f}`
- **Test Brier Score:** `{t_cal['brier_score']:.4f}` | **Test NLL:** `{t_cal['nll']:.4f}`

---

## 3. Selective Prediction & Uncertainty Abstention

Threshold $\\tau_{{\\text{{accept}}}}$ fitted on validation set: **{sel['tau_accept']:.4f}**

- **Coverage on Test Set:** **{sel['coverage'] * 100:.1f}%** ({sel['accepted_count']} / {sel['total_test_samples']})
- **Abstained (Uncertain):** **{sel['abstained_count']}** samples
- **Accuracy on Accepted Samples:** **{sel['accepted_accuracy']:.4f}** (Error rate: {sel['accepted_error_rate']:.4f})
- **Error Rate among Abstained Samples:** **{sel['abstained_error_rate']:.4f}**
- **False Negatives Caught by Abstention:** {err['test_fn_abstained']} out of {err['test_total_fn']} total FNs

---

## 4. Limitations & Disclaimers

"""
    for idx, lim in enumerate(report["limitations"], 1):
        md += f"{idx}. {lim}\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGuard Evaluation Suite")
    parser.add_argument("--basic", action="store_true", help="Run basic Phase 1 test evaluation")
    parser.add_argument("--full", action="store_true", help="Run full Phase 2+ validation report")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pth"))
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if args.basic:
        evaluate_basic(model_path=ckpt)
    elif args.full:
        evaluate_full(model_path=ckpt)
    else:
        # Default run full if checkpoint exists, otherwise basic
        if (MODELS_DIR / "thresholds.json").exists():
            evaluate_full(model_path=ckpt)
        else:
            evaluate_basic(model_path=ckpt)
