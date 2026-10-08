import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

from config import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_LOW_UNCERTAINTY_COVERAGE,
    DEFAULT_TARGET_ACCEPTED_ACC,
    DEFAULT_TARGET_LOW_UNCERTAINTY_ACC,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
)
from src.dataset import get_dataloaders
from src.evaluate import collect_predictions, compute_calibration_metrics
from src.model import load_checkpoint


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Applies temperature scaling and softmax to logits."""
    temp = max(1e-4, float(temperature))
    scaled = logits / temp
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def fit_temperature(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> Tuple[float, Dict]:
    """
    Finds the optimal temperature T that minimizes NLL on the validation logits.
    Uses scipy scalar bounded minimization over T in [0.1, 10.0].
    """
    def nll_at_temp(log_t: float) -> float:
        t = np.exp(log_t)
        probs = temperature_scale_logits(val_logits, t)
        return log_loss(val_labels, probs)

    # NLL before scaling (T=1)
    probs_raw = temperature_scale_logits(val_logits, 1.0)
    nll_before = float(log_loss(val_labels, probs_raw))

    result = minimize_scalar(
        nll_at_temp,
        bounds=(-2.3, 2.3),  # exp(-2.3)~0.1, exp(2.3)~10
        method="bounded",
        options={"xatol": 1e-6, "maxiter": 500},
    )

    optimal_temp = float(np.exp(result.x))
    probs_after = temperature_scale_logits(val_logits, optimal_temp)
    nll_after = float(log_loss(val_labels, probs_after))

    ece_before = compute_calibration_metrics(val_labels, probs_raw)["ece"]
    ece_after = compute_calibration_metrics(val_labels, probs_after)["ece"]

    print(f"[Calibration] Temperature: {optimal_temp:.4f} (T=1.0 NLL: {nll_before:.4f}, Scaled NLL: {nll_after:.4f})")
    print(f"[Calibration] Val ECE Before: {ece_before:.4f}, After: {ece_after:.4f}")

    meta = {
        "temperature": round(optimal_temp, 6),
        "method": "bounded_scalar_minimization_on_log_T",
        "fitted_on": "val",
        "val_nll_before": round(nll_before, 4),
        "val_nll_after": round(nll_after, 4),
        "val_ece_before": round(ece_before, 4),
        "val_ece_after": round(ece_after, 4),
        "n": int(len(val_labels)),
    }
    return optimal_temp, meta


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Computes Expected Calibration Error (confidence-based)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(in_bin):
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            prop_in_bin = np.mean(in_bin)
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    n_bins: int = 15,
) -> None:
    """Saves a reliability diagram showing calibration quality."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        count = np.sum(in_bin)
        bin_counts.append(int(count))
        if count > 0:
            bin_accs.append(float(np.mean(accuracies[in_bin])))
            bin_confs.append(float(np.mean(confidences[in_bin])))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float(bin_centers[i]))

    ece = compute_ece(probs, labels, n_bins)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, color="#4a90d9", alpha=0.8, label="Accuracy per bin", edgecolor="#2c5f8a")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def fit_abstention_thresholds(
    val_confs: np.ndarray,
    val_labels: np.ndarray,
    val_preds: np.ndarray,
    target_acc: float = DEFAULT_TARGET_ACCEPTED_ACC,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    target_low_acc: float = DEFAULT_TARGET_LOW_UNCERTAINTY_ACC,
    min_low_coverage: float = DEFAULT_MIN_LOW_UNCERTAINTY_COVERAGE,
) -> Dict:
    """
    Fits tau_accept and tau_low_uncertainty thresholds on validation confidence scores.

    tau_accept: smallest tau such that accepted-prediction accuracy >= target_acc
                AND coverage >= min_coverage.
    tau_low_uncertainty: same logic with stricter targets.
    If no tau satisfies both constraints, picks tau maximizing (accuracy * coverage).
    """
    n = len(val_labels)
    sorted_confs = np.sort(np.unique(val_confs))

    def scan_tau(t_acc: float, t_cov: float) -> Tuple[float, bool, float, float]:
        best_tau = sorted_confs[-1]
        best_score = -1.0
        found = False
        best_acc = 0.0
        best_cov = 0.0

        for candidate_tau in sorted_confs:
            accepted = val_confs >= candidate_tau
            coverage = float(np.mean(accepted))
            if coverage == 0:
                continue
            acc = float(np.mean(val_preds[accepted] == val_labels[accepted]))
            if acc >= t_acc and coverage >= t_cov:
                if not found:
                    found = True
                    best_tau = candidate_tau
                    best_acc = acc
                    best_cov = coverage
                break  # smallest tau satisfying constraints

        if not found:
            # fallback: maximize acc * coverage
            for candidate_tau in sorted_confs:
                accepted = val_confs >= candidate_tau
                coverage = float(np.mean(accepted))
                if coverage == 0:
                    continue
                acc = float(np.mean(val_preds[accepted] == val_labels[accepted]))
                score = acc * coverage
                if score > best_score:
                    best_score = score
                    best_tau = candidate_tau
                    best_acc = acc
                    best_cov = coverage

        return float(best_tau), not found, float(best_acc), float(best_cov)

    tau_accept, tau_accept_fallback, tau_accept_val_acc, tau_accept_val_cov = scan_tau(target_acc, min_coverage)
    tau_low, tau_low_fallback, tau_low_val_acc, tau_low_val_cov = scan_tau(target_low_acc, min_low_coverage)

    # Guarantee tau_low >= tau_accept
    if tau_low < tau_accept:
        tau_low = float(tau_accept + (1.0 - tau_accept) / 2.0)
        tau_low_fallback = True
        print(f"[Calibration] tau_low forced above tau_accept: {tau_low:.4f}")

    print(f"[Calibration] tau_accept = {tau_accept:.4f} (val acc={tau_accept_val_acc:.4f}, cov={tau_accept_val_cov:.4f}, fallback={tau_accept_fallback})")
    print(f"[Calibration] tau_low    = {tau_low:.4f} (val acc={tau_low_val_acc:.4f}, cov={tau_low_val_cov:.4f}, fallback={tau_low_fallback})")

    return {
        "tau_accept": round(float(tau_accept), 6),
        "tau_low_uncertainty": round(float(tau_low), 6),
        "decision_threshold": 0.5,
        "tau_accept_target_acc": target_acc,
        "tau_accept_min_coverage": min_coverage,
        "tau_accept_val_acc_achieved": round(tau_accept_val_acc, 4),
        "tau_accept_val_coverage_achieved": round(tau_accept_val_cov, 4),
        "tau_accept_fallback": tau_accept_fallback,
        "tau_low_target_acc": target_low_acc,
        "tau_low_min_coverage": min_low_coverage,
        "tau_low_val_acc_achieved": round(tau_low_val_acc, 4),
        "tau_low_val_coverage_achieved": round(tau_low_val_cov, 4),
        "tau_low_fallback": tau_low_fallback,
        "fitted_on": "val",
        "note": "decision_threshold=0.5 is a default, not clinically tuned.",
    }


def risk_coverage_curve(
    confs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    title: str,
    save_path: Path,
    tau_accept: float,
) -> None:
    """Saves risk (1-accuracy) vs coverage curve for the given split."""
    thresholds = np.sort(np.unique(confs))
    coverages = []
    errors = []

    for t in thresholds:
        accepted = confs >= t
        cov = float(np.mean(accepted))
        if cov == 0:
            continue
        err = 1.0 - float(np.mean(preds[accepted] == labels[accepted]))
        coverages.append(cov)
        errors.append(err)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, errors, color="#e74c3c", linewidth=2, label="Error Rate vs Coverage")
    ax.axvline(x=float(np.mean(confs >= tau_accept)), color="#2ecc71", linestyle="--", linewidth=1.5, label=f"tau_accept threshold")
    ax.set_xlabel("Coverage (Fraction Accepted)", fontsize=12)
    ax.set_ylabel("Error Rate (1 - Accuracy)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_calibration_fit(
    model_path: Path = MODELS_DIR / "best_model.pth",
    models_dir: Path = MODELS_DIR,
) -> None:
    """
    Full calibration pipeline:
    1. Collect val and test logits.
    2. Fit temperature on val.
    3. Fit abstention thresholds on val.
    4. Save calibration.json and thresholds.json.
    5. Generate reliability diagrams and risk-coverage curves.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_meta = load_checkpoint(model_path, device=device_str)

    dataloaders = get_dataloaders(batch_size=64, augment=False)
    val_logits, val_probs_raw, val_labels = collect_predictions(model, dataloaders["val"], torch.device(device_str))
    test_logits, test_probs_raw, test_labels = collect_predictions(model, dataloaders["test"], torch.device(device_str))

    # Fit temperature on val
    optimal_temp, calib_meta = fit_temperature(val_logits, val_labels)

    # Calibrated probabilities
    val_probs_cal = temperature_scale_logits(val_logits, optimal_temp)
    test_probs_cal = temperature_scale_logits(test_logits, optimal_temp)

    # Test ECE before/after
    test_ece_before = compute_ece(test_probs_raw, test_labels)
    test_ece_after = compute_ece(test_probs_cal, test_labels)
    calib_meta["test_ece_before"] = round(test_ece_before, 4)
    calib_meta["test_ece_after"] = round(test_ece_after, 4)
    print(f"[Calibration] Test  ECE Before: {test_ece_before:.4f}, After: {test_ece_after:.4f}")

    # Save calibration.json
    models_dir.mkdir(parents=True, exist_ok=True)
    calib_path = models_dir / "calibration.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)
    print(f"[Calibration] Saved calibration.json to {calib_path}")

    # Fit abstention thresholds on val
    val_confs = np.max(val_probs_cal, axis=1)
    val_preds = np.argmax(val_probs_cal, axis=1)
    thresh_data = fit_abstention_thresholds(val_confs, val_labels, val_preds)
    thresh_path = models_dir / "thresholds.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(thresh_data, f, indent=2)
    print(f"[Calibration] Saved thresholds.json to {thresh_path}")

    # Generate reliability diagrams
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reliability_diagram(val_probs_raw, val_labels, "Val Reliability (Before Calibration)",
                        REPORTS_DIR / "reliability_val_before.png")
    reliability_diagram(val_probs_cal, val_labels, "Val Reliability (After Temperature Scaling)",
                        REPORTS_DIR / "reliability_val_after.png")
    reliability_diagram(test_probs_raw, test_labels, "Test Reliability (Before Calibration)",
                        REPORTS_DIR / "reliability_test_before.png")
    reliability_diagram(test_probs_cal, test_labels, "Test Reliability (After Temperature Scaling)",
                        REPORTS_DIR / "reliability_test_after.png")

    # Risk-coverage curves
    tau_accept = thresh_data["tau_accept"]
    test_confs = np.max(test_probs_cal, axis=1)
    test_preds = np.argmax(test_probs_cal, axis=1)
    risk_coverage_curve(val_confs, val_labels, val_preds, "Risk-Coverage Curve (Val)",
                        REPORTS_DIR / "risk_coverage_val.png", tau_accept)
    risk_coverage_curve(test_confs, test_labels, test_preds, "Risk-Coverage Curve (Test)",
                        REPORTS_DIR / "risk_coverage_test.png", tau_accept)

    print("[Calibration] All calibration artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGuard AI Calibration Module")
    parser.add_argument("--fit", action="store_true", help="Fit and save temperature + thresholds")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pth"))
    args = parser.parse_args()

    if args.fit:
        run_calibration_fit(model_path=Path(args.checkpoint))
    else:
        print("Usage: python -m src.calibration --fit")
