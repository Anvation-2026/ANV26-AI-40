"""
python -m src.env_check
=======================
Environment validation for MedGuard AI ML module.
NO model is loaded. NO dataset is touched.
Exits with code 0 if all checks pass, 1 otherwise.

Usage:
    cd ml/
    python -m src.env_check
    python -m src.env_check --json      # machine-readable output
"""
import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

# Bootstrap path so config is importable
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("numpy", "NumPy"),
    ("cv2", "OpenCV (opencv-python)"),
    ("PIL", "Pillow"),
    ("sklearn", "scikit-learn"),
    ("scipy", "SciPy"),
    ("matplotlib", "Matplotlib"),
]

REQUIRED_ARTIFACTS = [
    "models/best_model.pth",
    "models/calibration.json",
    "models/thresholds.json",
    "models/quality_thresholds.json",
    "models/ood_stats.npz",
    "models/model_card.json",
    "reports/validation_report.json",
]

RECOMMENDED_ARTIFACTS = [
    "data/pneumoniamnist_224.npz",
]


def check_python() -> dict:
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major == 3 and minor >= 10)
    return {
        "name": "Python version",
        "value": f"{major}.{minor}.{sys.version_info.micro}",
        "ok": ok,
        "detail": "Requires Python 3.10+" if not ok else None,
    }


def check_package(import_name: str, display_name: str) -> dict:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "?")
        return {"name": display_name, "value": version, "ok": True, "detail": None}
    except ImportError as e:
        return {"name": display_name, "value": None, "ok": False, "detail": str(e)}


def check_torch_device() -> dict:
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            return {"name": "Compute device", "value": f"CUDA — {gpu}", "ok": True, "detail": None}
        return {"name": "Compute device", "value": "CPU (no CUDA)", "ok": True,
                "detail": "GPU recommended for speed, but CPU is fully supported."}
    except Exception as e:
        return {"name": "Compute device", "value": None, "ok": False, "detail": str(e)}


def check_artifact(rel_path: str, required: bool) -> dict:
    full = _ML_ROOT / rel_path
    exists = full.exists()
    return {
        "name": rel_path,
        "value": str(full) if exists else None,
        "ok": exists,
        "required": required,
        "detail": None if exists else f"Not found at {full}",
    }


def check_config_paths() -> dict:
    try:
        import config  # noqa: F401
        return {"name": "config.py imports", "value": "ok", "ok": True, "detail": None}
    except Exception as e:
        return {"name": "config.py imports", "value": None, "ok": False, "detail": str(e)}


def run_checks() -> dict:
    results = {
        "python": check_python(),
        "packages": [check_package(imp, name) for imp, name in REQUIRED_PACKAGES],
        "device": check_torch_device(),
        "config": check_config_paths(),
        "required_artifacts": [check_artifact(p, required=True) for p in REQUIRED_ARTIFACTS],
        "recommended_artifacts": [check_artifact(p, required=False) for p in RECOMMENDED_ARTIFACTS],
    }

    # Determine overall pass/fail
    failures = []
    if not results["python"]["ok"]:
        failures.append(results["python"]["name"])
    for pkg in results["packages"]:
        if not pkg["ok"]:
            failures.append(pkg["name"])
    if not results["device"]["ok"]:
        failures.append(results["device"]["name"])
    if not results["config"]["ok"]:
        failures.append(results["config"]["name"])
    for art in results["required_artifacts"]:
        if not art["ok"]:
            failures.append(art["name"])

    results["all_required_pass"] = len(failures) == 0
    results["failures"] = failures
    return results


def _print_results(results: dict) -> None:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    WARN = "\033[93mWARN\033[0m"

    print("\n=== MedGuard AI — Environment Check ===\n")

    # Python
    r = results["python"]
    print(f"  {'[PASS]' if r['ok'] else '[FAIL]'} {r['name']}: {r['value']}" +
          (f"  → {r['detail']}" if r['detail'] else ""))

    # Packages
    print("\n  --- Required Packages ---")
    for pkg in results["packages"]:
        status = "[PASS]" if pkg["ok"] else "[FAIL]"
        val = pkg["value"] or "(missing)"
        print(f"  {status} {pkg['name']}: {val}" +
              (f"  → {pkg['detail']}" if pkg["detail"] else ""))

    # Device
    r = results["device"]
    status = "[PASS]" if r["ok"] else "[FAIL]"
    print(f"\n  {status} {r['name']}: {r['value'] or '(error)'}" +
          (f"  → {r['detail']}" if r["detail"] else ""))

    # Config
    r = results["config"]
    status = "[PASS]" if r["ok"] else "[FAIL]"
    print(f"  {status} {r['name']}: {r['value'] or '(error)'}" +
          (f"  → {r['detail']}" if r["detail"] else ""))

    # Required artifacts
    print("\n  --- Required Model Artifacts ---")
    for art in results["required_artifacts"]:
        status = "[PASS]" if art["ok"] else "[FAIL]"
        print(f"  {status} {art['name']}" + (f"  → {art['detail']}" if art["detail"] else ""))

    # Recommended artifacts
    print("\n  --- Recommended (Optional) Artifacts ---")
    for art in results["recommended_artifacts"]:
        status = "[PASS]" if art["ok"] else "[WARN]"
        print(f"  {status} {art['name']}" + (f"  → {art['detail']}" if art["detail"] else ""))

    # Summary
    print()
    if results["all_required_pass"]:
        print("  [OK] All required checks PASSED. ML environment is ready.\n")
    else:
        print("  [ERR] FAILED checks:")
        for f in results["failures"]:
            print(f"      - {f}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MedGuard AI ML environment validation (no model loaded)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON to stdout"
    )
    args = parser.parse_args()

    results = run_checks()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_results(results)

    return 0 if results["all_required_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
