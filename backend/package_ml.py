"""
backend/package_ml.py
=====================
Automated packaging script for MedGuard AI.
Prepares the internal ML inference package inside backend/ for Vercel deployment,
packaging ONLY the necessary inference code and active model artifacts (~47.5 MB).

Leaves ml/ at the repository root as the single source of truth.
backend/ml is git-ignored and automatically generated during build.
"""
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
SOURCE_ML_DIR = REPO_ROOT / "ml"
TARGET_ML_DIR = BACKEND_DIR / "ml"

# Artifacts strictly required for inference & report APIs
REQUIRED_MODEL_FILES = [
    "best_model.pth",
    "ood_stats.npz",
    "calibration.json",
    "thresholds.json",
    "quality_thresholds.json",
    "artifacts_manifest.json",
    "model_card.json",
]


def package_ml():
    if not SOURCE_ML_DIR.exists():
        # If running in an already packaged environment where ml is already in backend
        if TARGET_ML_DIR.exists() and (TARGET_ML_DIR / "models" / "best_model.pth").exists():
            print("[package_ml] Target backend/ml already exists with required model artifacts. Skipping packaging.")
            return
        print(f"[package_ml] Error: Source ML directory not found at {SOURCE_ML_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"[package_ml] Packaging ML inference code from {SOURCE_ML_DIR} to {TARGET_ML_DIR}...")
    TARGET_ML_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Root python files
    for fname in ["__init__.py", "config.py", "interface.py"]:
        src = SOURCE_ML_DIR / fname
        if src.exists():
            shutil.copy2(src, TARGET_ML_DIR / fname)

    # 2. Source code directory (src/)
    target_src = TARGET_ML_DIR / "src"
    target_src.mkdir(parents=True, exist_ok=True)
    for py_file in (SOURCE_ML_DIR / "src").glob("*.py"):
        shutil.copy2(py_file, target_src / py_file.name)

    # 3. Model artifacts (models/) — strictly active inference weights
    target_models = TARGET_ML_DIR / "models"
    target_models.mkdir(parents=True, exist_ok=True)
    for m_file in REQUIRED_MODEL_FILES:
        src = SOURCE_ML_DIR / "models" / m_file
        if src.exists():
            shutil.copy2(src, target_models / m_file)
            print(f"  -> Packaged model artifact: {m_file} ({src.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"  -> [WARNING] Optional model file {m_file} not found at {src}")

    # 4. Reports (reports/) — validation_report.json only
    target_reports = TARGET_ML_DIR / "reports"
    target_reports.mkdir(parents=True, exist_ok=True)
    val_report = SOURCE_ML_DIR / "reports" / "validation_report.json"
    if val_report.exists():
        shutil.copy2(val_report, target_reports / "validation_report.json")

    print("[package_ml] Successfully packaged ML inference module into backend/ml.")


if __name__ == "__main__":
    package_ml()
