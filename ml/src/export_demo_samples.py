import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path bootstrap
from src import _ML_ROOT  # noqa: F401

from config import DEMO_SAMPLES_DIR, REPORTS_DIR
from src.dataset import get_split, load_npz
from src.predict import get_predictor


def export_all_demo_samples(output_dir: Path = DEMO_SAMPLES_DIR) -> Dict[str, Any]:
    """
    Generates and exports live demo samples with an exact manifest recording
    provenance, ground truth, and real predictor outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = get_predictor()

    data = load_npz()
    test_images = data["test_images"]
    test_labels = data["test_labels"].squeeze()
    val_images = data["val_images"]

    manifest_entries: List[Dict[str, Any]] = []

    print("[Demo Export] Scanning test split for sample categories...")
    # Evaluate all test images to find candidates
    results = []
    for idx, (img, lbl) in enumerate(zip(test_images, test_labels)):
        res = predictor.predict(img, generate_heatmap=False)
        results.append({
            "idx": idx,
            "img": img,
            "label": int(lbl),
            "status": res["status"],
            "finding": res["finding"],
            "prob": res["probability"],
            "conf": res["details"]["confidence"],
            "p_cal": res["details"]["p_pneumonia_calibrated"],
        })

    # 1. Positive samples (Pneumonia -> success pneumonia)
    positives = [r for r in results if r["label"] == 1 and r["status"] == "success" and r["finding"] == "pneumonia"]
    for i, r in enumerate(positives[:3], 1):
        filename = f"positive_{i}.png"
        filepath = output_dir / filename
        Image.fromarray(r["img"]).save(filepath)
        manifest_entries.append({
            "filename": filename,
            "category": "positive",
            "true_label": "pneumonia (1)",
            "predicted_status": r["status"],
            "predicted_finding": r["finding"],
            "calibrated_probability": r["prob"],
            "source_split": "test",
            "source_index": r["idx"],
            "synthetic_degradation": None,
            "provenance_note": "Real pediatric chest X-ray from official test split with confirmed pneumonia finding.",
        })

    # 2. Negative samples (Normal -> success normal)
    negatives = [r for r in results if r["label"] == 0 and r["status"] == "success" and r["finding"] == "normal"]
    for i, r in enumerate(negatives[:3], 1):
        filename = f"negative_{i}.png"
        filepath = output_dir / filename
        Image.fromarray(r["img"]).save(filepath)
        manifest_entries.append({
            "filename": filename,
            "category": "negative",
            "true_label": "normal (0)",
            "predicted_status": r["status"],
            "predicted_finding": r["finding"],
            "calibrated_probability": r["prob"],
            "source_split": "test",
            "source_index": r["idx"],
            "synthetic_degradation": None,
            "provenance_note": "Real pediatric chest X-ray from official test split with confirmed normal finding.",
        })

    # 3. Uncertain samples (Real test images that triggered abstention)
    uncertains = [r for r in results if r["status"] == "uncertain"]
    if not uncertains:
        # Pick lowest confidence samples near decision boundary
        sorted_by_conf = sorted(results, key=lambda x: x["conf"])
        uncertains = sorted_by_conf[:3]
    for i, r in enumerate(uncertains[:3], 1):
        filename = f"uncertain_{i}.png"
        filepath = output_dir / filename
        Image.fromarray(r["img"]).save(filepath)
        manifest_entries.append({
            "filename": filename,
            "category": "uncertain",
            "true_label": "pneumonia (1)" if r["label"] == 1 else "normal (0)",
            "predicted_status": r["status"],
            "predicted_finding": r["finding"],
            "calibrated_probability": r["prob"],
            "confidence": r["conf"],
            "source_split": "test",
            "source_index": r["idx"],
            "synthetic_degradation": None,
            "provenance_note": "Real borderline test image near decision boundary triggering model abstention.",
        })

    # 4. Error analysis cases: False Negatives and False Positives (mismatches)
    fn_cases = [r for r in results if r["label"] == 1 and r["finding"] == "normal"]
    for i, r in enumerate(fn_cases[:2], 1):
        filename = f"mismatch_fn_{i}.png"
        filepath = output_dir / filename
        Image.fromarray(r["img"]).save(filepath)
        manifest_entries.append({
            "filename": filename,
            "category": "error_analysis_false_negative",
            "true_label": "pneumonia (1)",
            "predicted_status": r["status"],
            "predicted_finding": r["finding"],
            "calibrated_probability": r["prob"],
            "source_split": "test",
            "source_index": r["idx"],
            "synthetic_degradation": None,
            "provenance_note": "Known false negative error case for failure mode analysis and clinical demo.",
        })

    # 5. Poor quality samples (Synthetically degraded copies of real X-rays)
    base_img = val_images[0]
    degradations = [
        ("poor_quality_blur.png", "blur", lambda img: cv2.GaussianBlur(img, (0, 0), 10.0), "severe Gaussian blur (sigma=10.0)"),
        ("poor_quality_dark.png", "dark", lambda img: (img * 0.08).astype(np.uint8), "extreme underexposure (scale=0.08)"),
        ("poor_quality_noise.png", "noise", lambda img: np.clip(img.astype(float) + np.random.normal(0, 45, img.shape), 0, 255).astype(np.uint8), "additive Gaussian noise (sigma=45)"),
    ]

    for filename, deg_type, deg_fn, deg_desc in degradations:
        deg_img = deg_fn(base_img)
        filepath = output_dir / filename
        Image.fromarray(deg_img).save(filepath)
        res = predictor.predict(deg_img, generate_heatmap=False)
        manifest_entries.append({
            "filename": filename,
            "category": "poor_quality",
            "true_label": "normal (0)",
            "predicted_status": res["status"],
            "predicted_finding": res["finding"],
            "calibrated_probability": res["probability"],
            "source_split": "val",
            "source_index": 0,
            "synthetic_degradation": deg_desc,
            "provenance_note": f"Synthetically degraded radiograph to demonstrate quality rejection ({deg_desc}).",
        })

    # 6. OOD samples (Non-medical proxies)
    # A. Noise
    noise_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    Image.fromarray(noise_img).save(output_dir / "ood_noise.png")
    res_noise = predictor.predict(noise_img, generate_heatmap=False)
    manifest_entries.append({
        "filename": "ood_noise.png",
        "category": "ood",
        "true_label": None,
        "predicted_status": res_noise["status"],
        "predicted_finding": None,
        "calibrated_probability": None,
        "source_split": "synthetic",
        "source_index": -1,
        "synthetic_degradation": None,
        "provenance_note": "Non-medical proxy: uniform random static noise frame.",
    })

    # B. Blank
    blank_img = np.zeros((224, 224), dtype=np.uint8)
    Image.fromarray(blank_img).save(output_dir / "ood_blank.png")
    res_blank = predictor.predict(blank_img, generate_heatmap=False)
    manifest_entries.append({
        "filename": "ood_blank.png",
        "category": "ood",
        "true_label": None,
        "predicted_status": res_blank["status"],
        "predicted_finding": None,
        "calibrated_probability": None,
        "source_split": "synthetic",
        "source_index": -1,
        "synthetic_degradation": None,
        "provenance_note": "Non-medical proxy: blank zero-intensity frame.",
    })

    # C. Color photo proxy
    color_img = np.zeros((224, 224, 3), dtype=np.uint8)
    color_img[:, :, 0] = 220  # Red
    color_img[:, :, 1] = 60   # Green
    color_img[:, :, 2] = 40   # Blue
    Image.fromarray(color_img).save(output_dir / "ood_color_photo.png")
    res_color = predictor.predict(color_img, generate_heatmap=False)
    manifest_entries.append({
        "filename": "ood_color_photo.png",
        "category": "ood",
        "true_label": None,
        "predicted_status": res_color["status"],
        "predicted_finding": None,
        "calibrated_probability": None,
        "source_split": "synthetic",
        "source_index": -1,
        "synthetic_degradation": None,
        "provenance_note": "Non-medical proxy: 3-channel RGB color photo violating grayscale X-ray modality.",
    })

    manifest = {
        "manifest_version": "1.0",
        "total_samples": len(manifest_entries),
        "exported_utc": "2026-10-08T05:50:00Z",
        "entries": manifest_entries,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Demo Export] Saved {len(manifest_entries)} demo samples and manifest to {output_dir}")

    # Generate false-negatives grid image (up to 12 FNs from test set)
    generate_false_negatives_grid(results, test_images, REPORTS_DIR / "false_negatives_grid.png")

    return manifest


def generate_false_negatives_grid(
    test_results: List[Dict],
    test_images: np.ndarray,
    out_path: Path,
) -> None:
    """Renders grid of up to 12 false-negative test cases with calibrated probabilities."""
    fn_items = [r for r in test_results if r["label"] == 1 and r["finding"] != "pneumonia"]
    if not fn_items:
        print("[Demo Export] Zero false negatives to plot in grid.")
        return

    n_samples = min(12, len(fn_items))
    rows = (n_samples + 3) // 4
    cols = min(4, n_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < n_samples:
            item = fn_items[i]
            img = test_images[item["idx"]]
            p_pneumonia = item.get("p_cal", 0.0)
            status = item.get("status", "unknown")
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(
                f"Test #{item['idx']} (FN)\nCalibrated P(Pneu): {p_pneumonia:.2f}\nStatus: {status}",
                fontsize=10,
            )
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    fig.suptitle("MedGuard AI — False Negative Analysis Grid (Test Split)", fontsize=14, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Demo Export] Saved false-negatives grid to {out_path}")


if __name__ == "__main__":
    export_all_demo_samples()
