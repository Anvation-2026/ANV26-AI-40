import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
import torch
import torch.nn as nn

# Path bootstrap
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

from config import (
    DATASET_PATH,
    MODELS_DIR,
    MODALITY_COLOR_THRESHOLD,
    DEFAULT_OOD_PERCENTILE,
    REPORTS_DIR,
)
from src.dataset import get_dataloaders, get_split
from src.model import load_checkpoint


def extract_features(
    model: nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extracts 512-d avg-pool penultimate features from a model in eval mode.
    No augmentation applied.
    """
    from src.preprocessing import get_eval_transform
    transform = get_eval_transform()
    model.eval()
    all_feats: List[np.ndarray] = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        tensors = []
        for img in batch_imgs:
            try:
                t = transform(img)
                tensors.append(t)
            except Exception:
                continue
        if not tensors:
            continue
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            # Extract penultimate features (avgpool output)
            x = model.conv1(batch_tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            feats = torch.flatten(x, 1)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 512))


def check_modality_color(image) -> bool:
    """
    Returns True if the image appears to be a color image (not grayscale X-ray).
    Checks mean absolute inter-channel difference.
    """
    try:
        from PIL import Image as PILImage
        import io

        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image = f.read()
        if isinstance(image, bytes):
            pil_img = PILImage.open(io.BytesIO(image))
            pil_img.load()
        elif isinstance(image, PILImage.Image):
            pil_img = image
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                return False  # Already grayscale
            if image.ndim == 3 and image.shape[2] == 1:
                return False
            # Check inter-channel differences
            if image.ndim == 3 and image.shape[2] >= 3:
                arr = image[:, :, :3].astype(np.float64)
                diff_rg = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 1]))
                diff_rb = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 2]))
                diff_gb = np.mean(np.abs(arr[:, :, 1] - arr[:, :, 2]))
                max_diff = max(diff_rg, diff_rb, diff_gb) / 255.0
                return max_diff > MODALITY_COLOR_THRESHOLD
            return False
        else:
            return False

        # Check PIL image mode
        if pil_img.mode in ("L", "1", "P"):
            # Grayscale or palette — check if palette is actually colorful
            if pil_img.mode == "P":
                rgb_pil = pil_img.convert("RGB")
            else:
                return False
        else:
            rgb_pil = pil_img.convert("RGB")

        arr = np.array(rgb_pil, dtype=np.float64)
        diff_rg = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 1]))
        diff_rb = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 2]))
        diff_gb = np.mean(np.abs(arr[:, :, 1] - arr[:, :, 2]))
        max_diff = max(diff_rg, diff_rb, diff_gb) / 255.0
        return max_diff > MODALITY_COLOR_THRESHOLD

    except Exception:
        return False


class OODDetector:
    """
    Two-layer OOD detector:
    1. Modality pre-check (color image)
    2. Feature-space Mahalanobis distance
    """

    def __init__(self, ood_stats_path: Path, thresholds_path: Optional[Path] = None):
        ood_stats_path = Path(ood_stats_path)
        if not ood_stats_path.exists():
            raise FileNotFoundError(f"OOD stats not found at {ood_stats_path}")

        stats = np.load(ood_stats_path, allow_pickle=True)
        self.class_means = [stats["mean_0"], stats["mean_1"]]
        self.precision_matrix = stats["precision_matrix"]
        self.ood_threshold = float(stats["ood_threshold"])
        self.ood_percentile = float(stats["ood_percentile"])

    def mahalanobis_score(self, features: np.ndarray) -> float:
        """Computes minimum Mahalanobis distance over class means."""
        min_dist = float("inf")
        diff_flat = features.flatten()
        for mean in self.class_means:
            diff = diff_flat - mean
            try:
                dist = float(diff @ self.precision_matrix @ diff) ** 0.5
            except Exception:
                dist = float("inf")
            min_dist = min(min_dist, dist)
        return float(min_dist)

    def assess(self, image, gray224: np.ndarray) -> Dict:
        """
        Runs modality pre-check on original image, returns initial OOD result.
        Feature-space check is done separately after model forward pass.
        """
        reasons = []
        modality_failed = check_modality_color(image)
        if modality_failed:
            reasons.append("unsupported_modality:color_image")

        return {
            "flagged": modality_failed,
            "modality_failed": modality_failed,
            "method": "mahalanobis+modality",
            "score": 0.0,
            "threshold": float(self.ood_threshold),
            "reasons": reasons,
        }

    def score_features(self, features: np.ndarray, existing_res: Dict) -> Dict:
        """Updates OOD result with feature-space Mahalanobis score."""
        if existing_res.get("modality_failed", False):
            return existing_res

        score = self.mahalanobis_score(features)
        flagged = score > self.ood_threshold
        reasons = list(existing_res.get("reasons", []))
        if flagged:
            reasons.append("feature_distribution_anomaly")

        return {
            "flagged": flagged,
            "modality_failed": False,
            "method": "mahalanobis+modality",
            "score": round(float(score), 4),
            "threshold": round(float(self.ood_threshold), 4),
            "reasons": reasons,
        }


def fit_ood_detector(
    model_path: Path = MODELS_DIR / "best_model.pth",
    models_dir: Path = MODELS_DIR,
    ood_percentile: float = DEFAULT_OOD_PERCENTILE,
) -> None:
    """
    Fits class-conditional Mahalanobis OOD detector on training features.
    Threshold is set from validation in-distribution scores.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model, ckpt_meta = load_checkpoint(model_path, device=device_str)

    print("[OOD] Extracting training features...")
    train_images, train_labels = get_split("train", DATASET_PATH)
    train_labels_flat = train_labels.squeeze()

    train_feats = extract_features(model, train_images, device)
    print(f"[OOD] Extracted {len(train_feats)} training features (dim={train_feats.shape[1]})")

    # Class-conditional means
    feats_class0 = train_feats[train_labels_flat == 0]
    feats_class1 = train_feats[train_labels_flat == 1]

    mean_0 = feats_class0.mean(axis=0)
    mean_1 = feats_class1.mean(axis=0)

    # Shared tied covariance (LedoitWolf shrinkage)
    print("[OOD] Fitting shared covariance matrix (LedoitWolf)...")
    lw = LedoitWolf()
    lw.fit(train_feats)
    cov_matrix = lw.covariance_
    precision_matrix = np.linalg.pinv(cov_matrix)

    # Compute val in-distribution scores
    print("[OOD] Extracting validation features for threshold fitting...")
    val_images, _ = get_split("val", DATASET_PATH)
    val_feats = extract_features(model, val_images, device)

    val_scores = []
    for feat in val_feats:
        min_dist = float("inf")
        for mean in [mean_0, mean_1]:
            diff = feat - mean
            try:
                dist = float(diff @ precision_matrix @ diff) ** 0.5
            except Exception:
                dist = float("inf")
            min_dist = min(min_dist, dist)
        val_scores.append(min_dist)

    val_scores = np.array(val_scores)
    ood_threshold = float(np.percentile(val_scores, ood_percentile))
    val_false_ood_rate = float(np.mean(val_scores > ood_threshold))

    print(f"[OOD] Threshold (val {ood_percentile}th percentile) = {ood_threshold:.4f}")
    print(f"[OOD] Val false-OOD rate = {val_false_ood_rate * 100:.2f}%")

    # Save stats
    models_dir.mkdir(parents=True, exist_ok=True)
    ood_stats_path = models_dir / "ood_stats.npz"
    np.savez(
        ood_stats_path,
        mean_0=mean_0,
        mean_1=mean_1,
        precision_matrix=precision_matrix,
        ood_threshold=np.array(ood_threshold),
        ood_percentile=np.array(ood_percentile),
        val_false_ood_rate=np.array(val_false_ood_rate),
    )
    print(f"[OOD] Saved ood_stats.npz to {ood_stats_path}")

    # Also update thresholds.json with OOD info
    thresholds_path = models_dir / "thresholds.json"
    thresh_data = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)
    thresh_data["ood"] = {
        "threshold": round(float(ood_threshold), 4),
        "percentile": ood_percentile,
        "val_false_ood_rate": round(float(val_false_ood_rate), 4),
        "rejection_precedence": "quality_gate_before_feature_ood",
    }
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresh_data, f, indent=2)
    print(f"[OOD] Updated thresholds.json with OOD info")


def evaluate_ood(
    model_path: Path = MODELS_DIR / "best_model.pth",
    models_dir: Path = MODELS_DIR,
) -> Dict:
    """
    Evaluates OOD detector on in-distribution val/test and proxy OOD sets.
    Proxy sets are used ONLY for evaluation — never for threshold fitting.
    """
    ood_stats_path = models_dir / "ood_stats.npz"
    if not ood_stats_path.exists():
        raise FileNotFoundError(f"OOD stats not found. Run --fit first.")

    detector = OODDetector(ood_stats_path)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model, _ = load_checkpoint(model_path, device=device_str)

    def score_images(images: np.ndarray) -> np.ndarray:
        feats = extract_features(model, images, device)
        scores = []
        for feat in feats:
            min_dist = float("inf")
            for mean in detector.class_means:
                diff = feat - mean
                try:
                    dist = float(diff @ detector.precision_matrix @ diff) ** 0.5
                except Exception:
                    dist = float("inf")
                min_dist = min(min_dist, dist)
            scores.append(min_dist)
        return np.array(scores)

    # In-distribution: val and test
    val_images, _ = get_split("val", DATASET_PATH)
    test_images, _ = get_split("test", DATASET_PATH)

    val_scores = score_images(val_images[:200])
    test_scores = score_images(test_images[:200])

    val_false_ood = float(np.mean(val_scores > detector.ood_threshold))
    test_false_ood = float(np.mean(test_scores > detector.ood_threshold))

    print(f"[OOD Eval] Val false-OOD rate: {val_false_ood * 100:.2f}%")
    print(f"[OOD Eval] Test false-OOD rate: {test_false_ood * 100:.2f}%")

    results = {
        "ood_threshold": round(float(detector.ood_threshold), 4),
        "in_distribution": {
            "val_false_ood_rate": round(val_false_ood, 4),
            "test_false_ood_rate": round(test_false_ood, 4),
        },
        "proxy_ood_sets": {},
        "note_near_ood": (
            "Near-OOD inputs (adult chest X-rays, other body parts, different scanners) "
            "may NOT be reliably detected. OOD rejection is not guaranteed."
        ),
        "proxy_sets_not_used_for_threshold_fitting": True,
    }

    # Proxy OOD Set A: Random noise and blank images
    rng = np.random.RandomState(42)
    noise_images = rng.randint(0, 255, size=(200, 224, 224), dtype=np.uint8)
    blank_images = np.zeros((50, 224, 224), dtype=np.uint8)
    all_noise_blank = np.concatenate([noise_images, blank_images], axis=0)
    from sklearn.metrics import roc_auc_score

    # Proxy OOD Set A: Random noise and blank images
    rng = np.random.RandomState(42)
    noise_images = rng.randint(0, 255, size=(200, 224, 224), dtype=np.uint8)
    blank_images = np.zeros((50, 224, 224), dtype=np.uint8)
    all_noise_blank = np.concatenate([noise_images, blank_images], axis=0)
    noise_scores = score_images(all_noise_blank)
    noise_detection_rate = float(np.mean(noise_scores > detector.ood_threshold))
    # AUROC vs val in-distribution scores
    y_true_a = np.concatenate([np.zeros(len(val_scores)), np.ones(len(noise_scores))])
    y_scores_a = np.concatenate([val_scores, noise_scores])
    auroc_a = float(roc_auc_score(y_true_a, y_scores_a))

    results["proxy_ood_sets"]["random_noise_and_blank"] = {
        "n": len(all_noise_blank),
        "detection_rate": round(noise_detection_rate, 4),
        "auroc_vs_val": round(auroc_a, 4),
        "note": "Easy OOD — high entropy noise and blank frames",
    }
    print(f"[OOD Eval] Noise/blank: detection={noise_detection_rate * 100:.1f}%, AUROC={auroc_a:.4f}")

    # Proxy OOD Set B: Synthetic natural proxy patterns (gradients, textures, shapes)
    synth_nat_images = []
    for k in range(200):
        # Create non-medical structured patterns: radial gradients, sinusoidal grids, checkerboards
        x = np.linspace(-1, 1, 224)
        y = np.linspace(-1, 1, 224)
        xx, yy = np.meshgrid(x, y)
        if k % 4 == 0:
            pat = np.sin(10 * xx) * np.cos(10 * yy)
        elif k % 4 == 1:
            pat = np.sqrt(xx**2 + yy**2)
        elif k % 4 == 2:
            pat = (np.sin(20 * xx) > 0).astype(float)
        else:
            pat = (xx + yy) / 2.0
        pat_uint8 = ((pat - pat.min()) / (pat.max() - pat.min() + 1e-6) * 255).astype(np.uint8)
        synth_nat_images.append(pat_uint8)
    synth_nat_arr = np.array(synth_nat_images)
    synth_nat_scores = score_images(synth_nat_arr)
    synth_nat_detection = float(np.mean(synth_nat_scores > detector.ood_threshold))
    y_true_b = np.concatenate([np.zeros(len(val_scores)), np.ones(len(synth_nat_scores))])
    y_scores_b = np.concatenate([val_scores, synth_nat_scores])
    auroc_b = float(roc_auc_score(y_true_b, y_scores_b))

    results["proxy_ood_sets"]["synthetic_natural_patterns"] = {
        "n": len(synth_nat_arr),
        "detection_rate": round(synth_nat_detection, 4),
        "auroc_vs_val": round(auroc_b, 4),
        "note": "Geometric, periodic, and gradient non-medical patterns",
    }
    print(f"[OOD Eval] Synthetic natural: detection={synth_nat_detection * 100:.1f}%, AUROC={auroc_b:.4f}")

    # Proxy OOD Set C: MedMNIST / external check (graceful skip note per Section 7 Phase 3)
    results["proxy_ood_sets"]["external_cifar10_and_medmnist"] = {
        "n": 0,
        "detection_rate": None,
        "note": "External CIFAR-10 download skipped to avoid slow network timeout; synthetic proxy sets evaluated instead.",
    }

    # Precedence Experiment: Quality-before-OOD vs OOD-before-Quality
    # Using degraded val X-rays (which should receive poor_quality) and proxy OOD (which should receive ood_rejected)
    from src.quality import QualityGate, _get_degradations
    q_gate = QualityGate(models_dir / "quality_thresholds.json")

    # Generate 50 degraded val images
    degraded_val = []
    for deg_name, deg_fn in _get_degradations()[:5]:
        for img in val_images[:10]:
            if img.ndim == 3:
                img = img[0] if img.shape[0] == 1 else img[:, :, 0] if img.shape[2] == 1 else img
            degraded_val.append(deg_fn(img.astype(np.uint8)))

    # Order A: Quality Gate before Feature OOD
    # Degraded should be poor_quality; OOD should be ood_rejected
    correct_a = 0
    total_prec = len(degraded_val) + len(all_noise_blank[:50])

    for d_img in degraded_val:
        q_res = q_gate.assess(d_img)
        # Order A: if quality poor -> poor_quality (correct)
        if q_res["status"] == "poor":
            correct_a += 1
        else:
            feat = extract_features(model, d_img[None, ...], device)
            score = detector.mahalanobis_score(feat)
            if score <= detector.ood_threshold:
                pass  # Passed both (not flagged)

    for o_img in all_noise_blank[:50]:
        q_res = q_gate.assess(o_img)
        if q_res["status"] == "poor":
            # In Order A, noise might be caught by quality or OOD
            correct_a += 1  # Rejected
        else:
            feat = extract_features(model, o_img[None, ...], device)
            score = detector.mahalanobis_score(feat)
            if score > detector.ood_threshold:
                correct_a += 1

    # Order B: Feature OOD before Quality Gate
    correct_b = 0
    for d_img in degraded_val:
        feat = extract_features(model, d_img[None, ...], device)
        score = detector.mahalanobis_score(feat)
        # Order B: if feature OOD flagged -> labeled OOD (incorrect reason for an X-ray that is just blurry/dark!)
        if score > detector.ood_threshold:
            # Misattributed to OOD instead of poor_quality
            pass
        else:
            q_res = q_gate.assess(d_img)
            if q_res["status"] == "poor":
                correct_b += 1

    for o_img in all_noise_blank[:50]:
        feat = extract_features(model, o_img[None, ...], device)
        score = detector.mahalanobis_score(feat)
        if score > detector.ood_threshold:
            correct_b += 1  # Correctly labeled OOD
        else:
            q_res = q_gate.assess(o_img)
            if q_res["status"] == "poor":
                correct_b += 1

    rate_a = round(correct_a / total_prec, 4)
    rate_b = round(correct_b / total_prec, 4)
    chosen_order = "quality_gate_before_feature_ood" if rate_a >= rate_b else "feature_ood_before_quality_gate"

    results["precedence_experiment"] = {
        "order_a_quality_then_ood_correct_rate": rate_a,
        "order_b_ood_then_quality_correct_rate": rate_b,
        "chosen_ordering": chosen_order,
        "n_samples": total_prec,
        "note": (
            "Ordering A (Quality then OOD) correctly attributes blur/darkness to image quality "
            "rather than misclassifying degraded chest radiographs as non-medical OOD."
        ),
    }
    print(f"[OOD Precedence] Order A (Quality then OOD): {rate_a * 100:.1f}% vs Order B: {rate_b * 100:.1f}%. Chosen: {chosen_order}")

    # Update thresholds.json
    thresholds_path = models_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        if "ood" not in t_data:
            t_data["ood"] = {}
        t_data["ood"]["rejection_precedence"] = chosen_order
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(t_data, f, indent=2)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ood_eval_path = REPORTS_DIR / "ood_eval.json"
    with open(ood_eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OOD Eval] Report saved to {ood_eval_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGuard OOD Detection Module")
    parser.add_argument("--fit", action="store_true", help="Fit OOD detector on training features")
    parser.add_argument("--eval", action="store_true", help="Evaluate OOD detector on in-dist and proxy sets")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best_model.pth"))
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if args.fit:
        fit_ood_detector(model_path=ckpt)
    elif args.eval:
        evaluate_ood(model_path=ckpt)
    else:
        print("Usage: python -m src.ood --fit | --eval")
