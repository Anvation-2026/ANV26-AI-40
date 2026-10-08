import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Path bootstrap
from src import _ML_ROOT  # noqa: F401

from config import (
    DATASET_PATH,
    MIN_RAW_IMAGE_DIM,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.dataset import get_split


# Metric functions
def compute_blur(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity normalized to [0, 1]."""
    return float(gray.mean() / 255.0)


def compute_contrast(gray: np.ndarray) -> float:
    """99th minus 1st percentile intensity in [0, 1]."""
    p1 = float(np.percentile(gray, 1)) / 255.0
    p99 = float(np.percentile(gray, 99)) / 255.0
    return float(p99 - p1)


def compute_noise_sigma(gray: np.ndarray) -> float:
    """Robust noise estimate: MAD of residual gray - medianBlur(gray, 3)."""
    median_filtered = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float64) - median_filtered.astype(np.float64)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    sigma = mad / 0.6745  # Consistent estimator of Gaussian sigma
    return float(sigma)


def compute_metrics(gray: np.ndarray) -> Dict[str, float]:
    """Computes all 4 quality metrics for a 224x224 uint8 gray image."""
    return {
        "blur_laplacian_var": round(compute_blur(gray), 2),
        "brightness_mean": round(compute_brightness(gray), 4),
        "contrast_p99_p1": round(compute_contrast(gray), 4),
        "noise_sigma": round(compute_noise_sigma(gray), 4),
    }


class QualityGate:
    """
    Assesses image quality against data-fitted thresholds.
    Loads thresholds from quality_thresholds.json artifact.
    """

    def __init__(self, thresholds_path: Path):
        thresholds_path = Path(thresholds_path)
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Quality thresholds not found at {thresholds_path}")
        with open(thresholds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.thresholds = data["thresholds"]

    def assess(self, gray224: np.ndarray) -> Dict:
        """
        Computes quality metrics and checks against fitted thresholds.
        Returns: {status: "acceptable"|"poor", metrics: {...}, reasons: [...]}
        """
        # Minimum raw-size check: here, gray224 is already 224x224
        # (raw-size check happens in predict.py before preprocessing)
        if gray224 is None or gray224.size == 0:
            return {"status": "poor", "metrics": {}, "reasons": ["corrupt"]}

        if gray224.std() < 0.5:
            return {"status": "poor", "metrics": {}, "reasons": ["corrupt"]}

        metrics = compute_metrics(gray224)
        reasons = []

        blur = metrics["blur_laplacian_var"]
        bright = metrics["brightness_mean"]
        contrast = metrics["contrast_p99_p1"]
        noise = metrics["noise_sigma"]

        if blur < self.thresholds.get("blur_min", 10.0):
            reasons.append("blur")
        if bright < self.thresholds.get("brightness_min", 0.05):
            reasons.append("too_dark")
        if bright > self.thresholds.get("brightness_max", 0.95):
            reasons.append("too_bright")
        if contrast < self.thresholds.get("contrast_min", 0.05):
            reasons.append("low_contrast")
        if noise > self.thresholds.get("noise_max", 50.0):
            reasons.append("noisy")

        status = "poor" if reasons else "acceptable"
        return {"status": status, "metrics": metrics, "reasons": reasons}


def fit_quality_thresholds(
    train_images: np.ndarray,
    val_images: np.ndarray,
) -> Dict:
    """
    Fits quality thresholds from training + validation clean images.
    Limits set at robust percentiles of clean-image distribution.
    False-rejection rate checked on validation-only images.
    """
    print(f"[Quality] Computing metrics on {len(train_images)} train + {len(val_images)} val images...")

    all_images = np.concatenate([train_images, val_images], axis=0)
    blurs, brights, contrasts, noises = [], [], [], []

    for i, img in enumerate(all_images):
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
            elif img.shape[2] == 1:
                img = img[:, :, 0]
        blurs.append(compute_blur(img))
        brights.append(compute_brightness(img))
        contrasts.append(compute_contrast(img))
        noises.append(compute_noise_sigma(img))
        if (i + 1) % 1000 == 0:
            print(f"[Quality] Processed {i+1}/{len(all_images)} images...")

    blurs = np.array(blurs)
    brights = np.array(brights)
    contrasts = np.array(contrasts)
    noises = np.array(noises)

    # Set limits at robust percentiles of clean images
    # Use 1st percentile for lower bounds (not p0.5 to be slightly lenient)
    thresholds = {
        "blur_min": float(np.percentile(blurs, 1.0)),
        "brightness_min": float(np.percentile(brights, 0.5)),
        "brightness_max": float(np.percentile(brights, 99.5)),
        "contrast_min": float(np.percentile(contrasts, 1.0)),
        "noise_max": float(np.percentile(noises, 99.5)),
    }

    stats = {
        "blur": {"mean": float(np.mean(blurs)), "std": float(np.std(blurs)),
                 "p1": float(np.percentile(blurs, 1)), "p99": float(np.percentile(blurs, 99))},
        "brightness": {"mean": float(np.mean(brights)), "std": float(np.std(brights)),
                       "p0.5": float(np.percentile(brights, 0.5)), "p99.5": float(np.percentile(brights, 99.5))},
        "contrast": {"mean": float(np.mean(contrasts)), "std": float(np.std(contrasts)),
                     "p1": float(np.percentile(contrasts, 1)), "p99": float(np.percentile(contrasts, 99))},
        "noise": {"mean": float(np.mean(noises)), "std": float(np.std(noises)),
                  "p0.5": float(np.percentile(noises, 0.5)), "p99.5": float(np.percentile(noises, 99.5))},
    }

    # Check false-rejection rate on validation images
    gate_proto = _apply_thresholds_vectorized(val_images, thresholds)
    false_rejection_rate_val = float(np.mean(gate_proto))
    print(f"[Quality] Val false-rejection rate: {false_rejection_rate_val*100:.2f}%")

    # If >10% rejected, relax to 0.5th percentile for lower bounds
    if false_rejection_rate_val > 0.10:
        thresholds["blur_min"] = float(np.percentile(blurs, 0.5))
        thresholds["contrast_min"] = float(np.percentile(contrasts, 0.5))
        gate_proto = _apply_thresholds_vectorized(val_images, thresholds)
        false_rejection_rate_val = float(np.mean(gate_proto))
        print(f"[Quality] After relaxing thresholds, val false-rejection rate: {false_rejection_rate_val*100:.2f}%")

    return {
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "percentiles_used": {
            "blur_min_pct": 1.0, "brightness_min_pct": 0.5, "brightness_max_pct": 99.5,
            "contrast_min_pct": 1.0, "noise_max_pct": 99.5,
        },
        "clean_image_stats": stats,
        "val_false_rejection_rate": round(false_rejection_rate_val, 4),
        "n_train": int(len(train_images)),
        "n_val": int(len(val_images)),
    }


def _apply_thresholds_vectorized(images: np.ndarray, thresholds: Dict) -> np.ndarray:
    """Returns boolean array: True = rejected."""
    rejected = np.zeros(len(images), dtype=bool)
    for i, img in enumerate(images):
        if img.ndim == 3:
            img = img[0] if img.shape[0] == 1 else img[:, :, 0] if img.shape[2] == 1 else img
        blur = compute_blur(img)
        bright = compute_brightness(img)
        contrast = compute_contrast(img)
        noise = compute_noise_sigma(img)
        if (blur < thresholds.get("blur_min", 10.0) or
            bright < thresholds.get("brightness_min", 0.05) or
            bright > thresholds.get("brightness_max", 0.95) or
            contrast < thresholds.get("contrast_min", 0.05) or
                noise > thresholds.get("noise_max", 50.0)):
            rejected[i] = True
    return rejected


def evaluate_quality(
    thresholds_path: Path,
    val_images: np.ndarray,
    test_images: np.ndarray,
) -> Dict:
    """Evaluates quality gate on val and test, and on synthetic degradations."""
    gate = QualityGate(thresholds_path)

    def eval_split(images: np.ndarray, split_name: str):
        results = []
        for img in images:
            if img.ndim == 3:
                img = img[0] if img.shape[0] == 1 else img[:, :, 0] if img.shape[2] == 1 else img
            res = gate.assess(img.astype(np.uint8))
            results.append(res["status"] == "poor")
        frr = float(np.mean(results))
        print(f"[Quality Eval] {split_name} false-rejection rate: {frr * 100:.2f}%")
        return round(frr, 4)

    val_frr = eval_split(val_images[:200], "Val")
    test_frr = eval_split(test_images[:200], "Test")

    # Synthetic degradation evaluation on a subset of val
    subset = val_images[:200]
    degradation_results = {}

    for deg_name, deg_fn in _get_degradations():
        n_detected = 0
        for img in subset:
            if img.ndim == 3:
                img = img[0] if img.shape[0] == 1 else img[:, :, 0] if img.shape[2] == 1 else img
            degraded = deg_fn(img.astype(np.uint8))
            res = gate.assess(degraded)
            if res["status"] == "poor":
                n_detected += 1
        det_rate = round(n_detected / len(subset), 4)
        degradation_results[deg_name] = det_rate
        print(f"[Quality Eval] {deg_name}: detection rate = {det_rate * 100:.1f}%")

    return {
        "val_false_rejection_rate": val_frr,
        "test_false_rejection_rate": test_frr,
        "synthetic_degradation_detection_rates": degradation_results,
        "note": "Degraded samples are synthetic; labeled as such. Mild degradations may legitimately pass.",
    }


def _get_degradations():
    """Returns list of (name, degradation_function) pairs for evaluation."""
    import cv2 as cv2_inner
    return [
        ("gaussian_blur_sigma3", lambda img: cv2_inner.GaussianBlur(img, (0, 0), 3.0)),
        ("gaussian_blur_sigma8", lambda img: cv2_inner.GaussianBlur(img, (0, 0), 8.0)),
        ("gaussian_blur_sigma15", lambda img: cv2_inner.GaussianBlur(img, (0, 0), 15.0)),
        ("darkening_gamma0.2", lambda img: (np.clip(img.astype(np.float32) / 255.0, 0, 1) ** 0.2 * 255).astype(np.uint8)),
        ("darkening_scale0.1", lambda img: np.clip(img.astype(np.float32) * 0.1, 0, 255).astype(np.uint8)),
        ("overexposure_scale3", lambda img: np.clip(img.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)),
        ("gaussian_noise_sigma25", lambda img: np.clip(img.astype(np.float32) + np.random.normal(0, 25, img.shape), 0, 255).astype(np.uint8)),
        ("gaussian_noise_sigma50", lambda img: np.clip(img.astype(np.float32) + np.random.normal(0, 50, img.shape), 0, 255).astype(np.uint8)),
        ("contrast_squash", lambda img: np.clip(img.astype(np.float32) * 0.3 + 100, 0, 255).astype(np.uint8)),
        ("jpeg_quality5", lambda img: _jpeg_degrade(img, quality=5)),
    ]


def _jpeg_degrade(gray: np.ndarray, quality: int = 5) -> np.ndarray:
    """Degrades image by JPEG compression at low quality."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", gray, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def assess_quality(gray224: np.ndarray) -> Dict:
    """
    Public convenience function to assess quality with fitted thresholds.
    Returns raw metrics dict even if QualityGate artifact is unavailable.
    """
    thresholds_path = MODELS_DIR / "quality_thresholds.json"
    if not thresholds_path.exists():
        metrics = compute_metrics(gray224)
        return {"status": "acceptable", "metrics": metrics, "reasons": [], "note": "No fitted thresholds; using raw metrics only"}
    gate = QualityGate(thresholds_path)
    return gate.assess(gray224)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGuard Image Quality Module")
    parser.add_argument("--fit", action="store_true", help="Fit quality thresholds from dataset")
    parser.add_argument("--eval", action="store_true", help="Evaluate quality gate on val/test and synthetic degradations")
    args = parser.parse_args()

    if args.fit:
        from src.dataset import get_split
        print("[Quality] Loading training and validation images...")
        train_images, _ = get_split("train", DATASET_PATH)
        val_images, _ = get_split("val", DATASET_PATH)

        quality_data = fit_quality_thresholds(train_images, val_images)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODELS_DIR / "quality_thresholds.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(quality_data, f, indent=2)
        print(f"[Quality] Saved quality_thresholds.json to {out_path}")

    elif args.eval:
        thresholds_path = MODELS_DIR / "quality_thresholds.json"
        if not thresholds_path.exists():
            print("[Quality] quality_thresholds.json not found. Run --fit first.")
        else:
            from src.dataset import get_split
            val_images, _ = get_split("val", DATASET_PATH)
            test_images, _ = get_split("test", DATASET_PATH)
            eval_result = evaluate_quality(thresholds_path, val_images, test_images)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = REPORTS_DIR / "quality_eval.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2)
            print(f"[Quality] Evaluation report saved to {out_path}")
    else:
        print("Usage: python -m src.quality --fit | --eval")
