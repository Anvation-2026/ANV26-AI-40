"""
contract_check.py — MedGuard AI Phase 6 Regression + Contract Validator
=======================================================================
Verifies that every response from predict_image():
  1. Contains all required top-level and detail keys.
  2. Has correct types for every key.
  3. No required key is null unless the contract explicitly allows it.
  4. details.quality contains thresholds, passed, and labels dicts.
  5. Heatmap metadata keys (heatmap_target_class, heatmap_note) are present.

Uses demo_samples/ images so NO dataset download is required.
Exits 0 if all assertions pass, 1 on any failure.

Usage:
    cd ml/
    python contract_check.py
    python contract_check.py --json
"""
import argparse
import json
import sys
from pathlib import Path

# Bootstrap path
_ML_ROOT = Path(__file__).resolve().parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from src.predict import get_predictor, model_status, predict_image  # noqa: E402

# ─── Contract Definition ─────────────────────────────────────────────────────

TOP_LEVEL_KEYS = {
    "status": str,
    "finding": (str, type(None)),
    "probability": (float, type(None)),
    "uncertainty": (str, type(None)),
    "quality": (str, type(None)),
    "ood": (bool, type(None)),
    "explanation": str,
    "evidence": list,
    "heatmap_path": (str, type(None)),
    "model_version": str,
    "limitations": list,
    "details": dict,
}

DETAILS_KEYS = {
    "p_pneumonia_raw": (float, type(None)),
    "p_pneumonia_calibrated": (float, type(None)),
    "confidence": (float, type(None)),
    "uncertainty_score": (float, type(None)),
    "temperature": (float, type(None)),
    "calibrated": bool,
    "thresholds": dict,
    "quality": (dict, type(None)),
    "ood": (dict, type(None)),
    "rejection_reason": (str, type(None)),
    "inference_ms": float,
    "device": str,
    "capabilities": dict,
    # Task 5 keys
    "heatmap_target_class": (str, type(None)),
    "heatmap_note": (str, type(None)),
}

# Keys that must NOT be null when status == "success"
REQUIRED_NON_NULL_ON_SUCCESS = [
    "finding", "probability", "uncertainty", "quality", "ood",
]

VALID_STATUSES = {"success", "uncertain", "poor_quality", "ood_rejected", "unsupported_file"}
VALID_FINDINGS = {"normal", "pneumonia", None}
VALID_UNCERTAINTY = {"low", "medium", "high", None}

# Task 1: quality sub-dict expected keys
QUALITY_EXPECTED_KEYS = {"status", "metrics", "thresholds", "passed", "labels", "reasons"}

# ─── Model Status Contract ────────────────────────────────────────────────────

MODEL_STATUS_KEYS = {
    "available": bool,
    "model_name": (str, type(None)),
    "model_version": str,
    "dataset": (str, type(None)),
    "classes": list,
    "supported_modality": str,
    "device": str,
    "gpu_name": str,
    "capabilities": dict,
    "validation_report_available": bool,
    "trained_on": str,
    "error": (str, type(None)),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _type_ok(value, expected_type) -> bool:
    if isinstance(expected_type, tuple):
        return isinstance(value, expected_type)
    return isinstance(value, expected_type)


def check_response(result: dict, image_label: str, issues: list) -> None:
    """Validates a single predict() response against the full contract."""

    # 1. Top-level keys and types
    for key, expected_type in TOP_LEVEL_KEYS.items():
        if key not in result:
            issues.append(f"[{image_label}] Missing top-level key: '{key}'")
            continue
        if not _type_ok(result[key], expected_type):
            issues.append(
                f"[{image_label}] '{key}' expected {expected_type}, got {type(result[key]).__name__}={result[key]!r}"
            )

    # 2. Status is valid
    status = result.get("status")
    if status not in VALID_STATUSES:
        issues.append(f"[{image_label}] Invalid status: {status!r}")

    # 3. Non-null enforcement for success
    if status == "success":
        for key in REQUIRED_NON_NULL_ON_SUCCESS:
            if result.get(key) is None:
                issues.append(f"[{image_label}] status=success but '{key}' is null")
        if result.get("finding") not in VALID_FINDINGS:
            issues.append(f"[{image_label}] Invalid finding: {result.get('finding')!r}")
        if result.get("uncertainty") not in VALID_UNCERTAINTY:
            issues.append(f"[{image_label}] Invalid uncertainty: {result.get('uncertainty')!r}")

    # 4. Details keys and types
    details = result.get("details", {})
    if not isinstance(details, dict):
        issues.append(f"[{image_label}] 'details' is not a dict")
        return

    for key, expected_type in DETAILS_KEYS.items():
        if key not in details:
            issues.append(f"[{image_label}] Missing details key: '{key}'")
            continue
        if not _type_ok(details[key], expected_type):
            issues.append(
                f"[{image_label}] details['{key}'] expected {expected_type}, got "
                f"{type(details[key]).__name__}={details[key]!r}"
            )

    # 5. Task 1: quality sub-dict structure (when not None)
    quality_detail = details.get("quality")
    if quality_detail is not None:
        missing_q = QUALITY_EXPECTED_KEYS - set(quality_detail.keys())
        if missing_q:
            issues.append(f"[{image_label}] details.quality missing keys: {missing_q}")
        # passed and thresholds must be dicts when quality is present
        for sub in ("passed", "thresholds"):
            v = quality_detail.get(sub)
            if v is not None and not isinstance(v, dict):
                issues.append(f"[{image_label}] details.quality.{sub} expected dict, got {type(v).__name__}")

    # 6. JSON serializable
    try:
        json.dumps(result)
    except (TypeError, ValueError) as e:
        issues.append(f"[{image_label}] Result is not JSON-serializable: {e}")


def load_demo_images() -> list:
    """Loads demo images from demo_samples/. Falls back to synthetic if empty."""
    demo_dir = _ML_ROOT / "demo_samples"
    images = []
    if demo_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in sorted(demo_dir.glob(ext)):
                images.append(("demo:" + p.name, p))
    if not images:
        # Synthetic fallback: create a gray 224x224 image as bytes
        import numpy as np
        import cv2
        synthetic = np.random.randint(40, 200, (224, 224), dtype=np.uint8)
        _, buf = cv2.imencode(".png", synthetic)
        images.append(("synthetic:gray224", buf.tobytes()))
    return images


def run_contract_check(verbose: bool = False) -> dict:
    results = {
        "checks_run": 0,
        "checks_passed": 0,
        "checks_failed": 0,
        "issues": [],
        "images_tested": [],
        "model_status_issues": [],
        "overall_pass": False,
    }

    # ── model_status() contract ──────────────────────────────────────────────
    ms = model_status()
    for key, expected_type in MODEL_STATUS_KEYS.items():
        results["checks_run"] += 1
        if key not in ms:
            msg = f"[model_status] Missing key: '{key}'"
            results["issues"].append(msg)
            results["model_status_issues"].append(msg)
            results["checks_failed"] += 1
        elif not _type_ok(ms[key], expected_type):
            msg = (f"[model_status] '{key}' expected {expected_type}, "
                   f"got {type(ms[key]).__name__}={ms[key]!r}")
            results["issues"].append(msg)
            results["model_status_issues"].append(msg)
            results["checks_failed"] += 1
        else:
            results["checks_passed"] += 1

    # ── Per-image contract checks ─────────────────────────────────────────────
    images = load_demo_images()
    for label, img in images:
        issues_before = len(results["issues"])
        try:
            res = predict_image(img, generate_heatmap=False)
            check_response(res, label, results["issues"])
        except Exception as e:
            results["issues"].append(f"[{label}] predict_image() raised: {e}")

        new_issues = results["issues"][issues_before:]
        n_new = len(new_issues)
        results["checks_run"] += 1
        if n_new == 0:
            results["checks_passed"] += 1
            results["images_tested"].append({"image": label, "status": res.get("status"), "pass": True})
        else:
            results["checks_failed"] += 1
            results["images_tested"].append({"image": label, "status": res.get("status", "?"), "pass": False,
                                             "issues": new_issues})
        if verbose:
            status_str = "PASS" if n_new == 0 else "FAIL"
            print(f"  [{status_str}] {label}: status={res.get('status', '?')}")
            for iss in new_issues:
                print(f"          {iss}")

    results["overall_pass"] = results["checks_failed"] == 0
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MedGuard AI Phase 6 contract regression check."
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-image details")
    args = parser.parse_args()

    print("\n=== MedGuard AI — Contract Check (Phase 6 Regression) ===\n")

    try:
        results = run_contract_check(verbose=args.verbose or not args.json)
    except Exception as e:
        print(f"FATAL: contract_check failed to run: {e}")
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n  Checks run:    {results['checks_run']}")
        print(f"  Checks passed: {results['checks_passed']}")
        print(f"  Checks failed: {results['checks_failed']}")
        if results["issues"]:
            print("\n  ISSUES FOUND:")
            for iss in results["issues"]:
                print(f"    - {iss}")
        verdict = "PASS" if results["overall_pass"] else "FAIL"
        print(f"\n  Overall: {verdict}\n")

    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
