import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

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
            if hasattr(model, "get_2d_logits"):
                logits = model.get_2d_logits(inputs)
            else:
                logits = model(inputs)
                if logits.shape[1] == 1:
                    logits = torch.cat([-logits / 2.0, logits / 2.0], dim=1)
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

    # Bootstrap CIs (1000 resamples)
    test_boot_cis = compute_bootstrap_cis(test_targets, test_probs_calib[:, 1])
    val_boot_cis = compute_bootstrap_cis(val_targets, val_probs_calib[:, 1])

    # Load thresholds for selective prediction
    thresholds_file = models_dir / "thresholds.json"
    thresh_meta = {}
    if thresholds_file.exists():
        with open(thresholds_file, "r", encoding="utf-8") as f:
            thresh_meta = json.load(f)

    def format_metrics_block(
        b_metrics: Dict[str, Any],
        c_metrics: Dict[str, Any],
        ci_src: Optional[Dict[str, Any]],
        sample_count: int,
    ) -> Dict[str, Any]:
        formatted_ci = {}
        for metric_name in ["accuracy", "sensitivity", "specificity", "auroc"]:
            if ci_src and metric_name in ci_src:
                entry = ci_src[metric_name]
                if isinstance(entry, dict):
                    formatted_ci[metric_name] = [
                        round(entry.get("ci_lower", 0.0), 4),
                        round(entry.get("ci_upper", 0.0), 4),
                    ]
                elif isinstance(entry, list) and len(entry) == 2:
                    formatted_ci[metric_name] = [round(entry[0], 4), round(entry[1], 4)]
                else:
                    formatted_ci[metric_name] = [round(b_metrics[metric_name], 4), round(b_metrics[metric_name], 4)]
            else:
                val = round(b_metrics[metric_name], 4)
                formatted_ci[metric_name] = [val, val]

        return {
            "n": sample_count,
            "accuracy": round(b_metrics["accuracy"], 4),
            "precision": round(b_metrics["precision"], 4),
            "recall": round(b_metrics["sensitivity"], 4),
            "sensitivity": round(b_metrics["sensitivity"], 4),
            "specificity": round(b_metrics["specificity"], 4),
            "f1": round(b_metrics["f1"], 4),
            "auroc": round(b_metrics["auroc"], 4),
            "false_negative_rate": round(b_metrics["false_negative_rate"], 4),
            "false_positive_rate": round(b_metrics["false_positive_rate"], 4),
            "ece": round(c_metrics["ece"], 4),
            "brier": round(c_metrics["brier_score"], 4),
            "nll": round(c_metrics["nll"], 4),
            "confusion_matrix": {
                "labels": ["normal", "pneumonia"],
                "matrix": b_metrics["confusion_matrix"],
            },
            "ci": formatted_ci,
        }

    def compute_selective_block(
        p_cal: np.ndarray,
        y_true: np.ndarray,
        t_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        tau_acc = float(t_meta.get("tau_accept", 0.80))
        tau_l = float(t_meta.get("tau_low_uncertainty", 0.95))
        c_vals = np.max(p_cal, axis=1)
        acc_mask = c_vals >= tau_acc
        abs_mask = ~acc_mask
        n_tot = len(y_true)
        acc_cnt = int(np.sum(acc_mask))
        abs_cnt = int(np.sum(abs_mask))
        cov = float(acc_cnt / n_tot) if n_tot > 0 else 0.0

        if acc_cnt > 0:
            s_acc = compute_binary_metrics(y_true[acc_mask], p_cal[acc_mask, 1])
        else:
            s_acc = {"accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}

        return {
            "tau_accept": round(tau_acc, 6),
            "tau_low_uncertainty": round(tau_l, 6),
            "coverage": round(cov, 4),
            "accepted_accuracy": round(s_acc["accuracy"], 4),
            "accepted_sensitivity": round(s_acc["sensitivity"], 4),
            "accepted_specificity": round(s_acc["specificity"], 4),
            "abstained_count": abs_cnt,
            "fallback": bool(t_meta.get("tau_accept_fallback", False)),
        }

    # Selective blocks
    selective_test = compute_selective_block(test_probs_calib, test_targets, thresh_meta)
    selective_val = compute_selective_block(val_probs_calib, val_targets, thresh_meta)

    # Error analysis: False Negatives and False Positives on test
    tau_acc_test = selective_test["tau_accept"]
    test_confs = np.max(test_probs_calib, axis=1)
    abstained_mask = test_confs < tau_acc_test
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
    quality_eval = {}
    quality_eval_file = REPORTS_DIR / "quality_eval.json"
    if quality_eval_file.exists():
        with open(quality_eval_file, "r", encoding="utf-8") as f:
            quality_eval = json.load(f)

    ood_eval = {}
    ood_eval_file = REPORTS_DIR / "ood_eval.json"
    if ood_eval_file.exists():
        with open(ood_eval_file, "r", encoding="utf-8") as f:
            ood_eval = json.load(f)

    report_json = {
        "schema_version": "1.0",
        "provenance": {
            "model_version": ckpt_meta.get("full_model_version", "unknown"),
            "checkpoint_sha256": ckpt_meta.get("sha256", "unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "device": str(device),
            "split_sizes": ds_meta["split_sizes"],
        },
        "dataset": {
            "name": ds_meta["dataset_name"],
            "image_size": 224,
            "class_names": ["normal", "pneumonia"],
            "class_counts": {
                "train": {
                    "normal": ds_meta["class_counts"]["train"]["normal (0)"],
                    "pneumonia": ds_meta["class_counts"]["train"]["pneumonia (1)"],
                },
                "val": {
                    "normal": ds_meta["class_counts"]["val"]["normal (0)"],
                    "pneumonia": ds_meta["class_counts"]["val"]["pneumonia (1)"],
                },
                "test": {
                    "normal": ds_meta["class_counts"]["test"]["normal (0)"],
                    "pneumonia": ds_meta["class_counts"]["test"]["pneumonia (1)"],
                },
            },
            "duplicate_overlap": ds_meta.get("exact_duplicates", {}),
        },
        "metrics": {
            "test": {
                "uncalibrated": format_metrics_block(test_raw_metrics, test_calib_err_raw, test_boot_cis, len(test_targets)),
                "calibrated": format_metrics_block(test_calib_metrics, test_calib_err_cal, test_boot_cis, len(test_targets)),
            },
            "val": {
                "uncalibrated": format_metrics_block(val_raw_metrics, val_calib_err_raw, val_boot_cis, len(val_targets)),
                "calibrated": format_metrics_block(val_calib_metrics, val_calib_err_cal, val_boot_cis, len(val_targets)),
            },
        },
        "calibration": {
            "method": "temperature_scaling",
            "temperature": round(temperature, 4),
            "fitted_on": "val",
            "val": {
                "ece_before": round(val_calib_err_raw["ece"], 4),
                "ece_after": round(val_calib_err_cal["ece"], 4),
            },
            "test": {
                "ece_before": round(test_calib_err_raw["ece"], 4),
                "ece_after": round(test_calib_err_cal["ece"], 4),
            },
        },
        "selective": {
            "test": selective_test,
            "val": selective_val,
        },
        "quality_eval": quality_eval,
        "ood_eval": ood_eval,
        "error_analysis": error_analysis,
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
    sel = report["selective"]["test"]
    cal = report["calibration"]
    err = report["error_analysis"]

    ci = t_cal.get("ci", {})

    md = f"""# MedGuard AI — Model Validation Report

**Schema Version:** `{report.get('schema_version', '1.0')}`  
**Model Version:** `{prov['model_version']}`  
**Generated At:** `{prov.get('generated_at', '')}`  
**Hardware Device:** `{prov['device']}`  
**Dataset:** {report['dataset']['name']} ({report['dataset']['image_size']}x{report['dataset']['image_size']})

---

## 1. Executive Summary & Test Performance

Primary evaluation on official **Test split** (N = {prov['split_sizes']['test']}):

| Metric | Test Value (Calibrated) | 95% Bootstrap CI | Validation Value |
|---|---|---|---|
| **AUROC** | **{t_cal['auroc']:.4f}** | [{ci.get('auroc', [0, 0])[0]:.4f}, {ci.get('auroc', [0, 0])[1]:.4f}] | {v_cal['auroc']:.4f} |
| **Accuracy** | **{t_cal['accuracy']:.4f}** | [{ci.get('accuracy', [0, 0])[0]:.4f}, {ci.get('accuracy', [0, 0])[1]:.4f}] | {v_cal['accuracy']:.4f} |
| **Sensitivity (Recall)** | **{t_cal['sensitivity']:.4f}** | [{ci.get('sensitivity', [0, 0])[0]:.4f}, {ci.get('sensitivity', [0, 0])[1]:.4f}] | {v_cal['sensitivity']:.4f} |
| **Specificity** | **{t_cal['specificity']:.4f}** | [{ci.get('specificity', [0, 0])[0]:.4f}, {ci.get('specificity', [0, 0])[1]:.4f}] | {v_cal['specificity']:.4f} |
| **Precision** | **{t_cal['precision']:.4f}** | - | {v_cal['precision']:.4f} |
| **F1 Score** | **{t_cal['f1']:.4f}** | - | {v_cal['f1']:.4f} |
| **False-Negative Rate** | **{t_cal['false_negative_rate']:.4f}** | - | {v_cal['false_negative_rate']:.4f} |

### Confusion Matrix (Test Split)
```
                Predicted Normal    Predicted Pneumonia
Actual Normal        {t_cal['confusion_matrix']['matrix'][0][0]:<19} {t_cal['confusion_matrix']['matrix'][0][1]}
Actual Pneumonia     {t_cal['confusion_matrix']['matrix'][1][0]:<19} {t_cal['confusion_matrix']['matrix'][1][1]}
```

---

## 2. Calibration & Temperature Scaling

Fitted on **Validation Split** (N = {prov['split_sizes']['val']}):
- **Learned Temperature $T$:** `{cal['temperature']:.4f}`
- **Validation ECE:** `{cal['val']['ece_before']:.4f}` $\\to$ `{cal['val']['ece_after']:.4f}`
- **Test ECE:** `{cal['test']['ece_before']:.4f}` $\\to$ `{cal['test']['ece_after']:.4f}`
- **Test Brier Score:** `{t_cal['brier']:.4f}` | **Test NLL:** `{t_cal['nll']:.4f}`

---

## 3. Selective Prediction & Uncertainty Abstention

Threshold $\\tau_{{\\text{{accept}}}}$ fitted on validation set: **{sel['tau_accept']:.4f}**

- **Coverage on Test Set:** **{sel['coverage'] * 100:.1f}%**
- **Abstained (Uncertain):** **{sel['abstained_count']}** samples
- **Accuracy on Accepted Samples:** **{sel['accepted_accuracy']:.4f}**
- **False Negatives Caught by Abstention:** {err.get('test_fn_abstained', 0)} out of {err.get('test_total_fn', 0)} total FNs

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
