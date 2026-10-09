"""
MedGuard AI — Model Weights Splitter & Merger Utility
Handles GitHub's 100MB file size limit by splitting large model checkpoints
into safe ~45MB binary chunks (.part00, .part01, ...) and reassembling them.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Optional

CHUNK_SIZE = 45 * 1024 * 1024  # 45 MB per chunk (well below GitHub's 100MB limit and 50MB warning)
MODELS_DIR = Path(__file__).resolve().parent


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def split_file(filepath: Path, chunk_size: int = CHUNK_SIZE) -> List[Path]:
    """Split a large file into chunks < 50MB."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = filepath.stat().st_size
    if file_size <= chunk_size:
        print(f"Skipping {filepath.name}: size {file_size / (1024*1024):.2f}MB <= chunk size.")
        return []

    print(f"Splitting {filepath.name} ({file_size / (1024*1024):.2f} MB)...")
    original_sha256 = compute_sha256(filepath)

    parts: List[Path] = []
    part_idx = 0
    with open(filepath, "rb") as f_in:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            part_name = filepath.parent / f"{filepath.name}.part{part_idx:02d}"
            with open(part_name, "wb") as f_out:
                f_out.write(chunk)
            parts.append(part_name)
            print(f"  Created part {part_idx:02d}: {part_name.name} ({len(chunk) / (1024*1024):.2f} MB)")
            part_idx += 1

    manifest = {
        "original_filename": filepath.name,
        "total_size_bytes": file_size,
        "total_parts": len(parts),
        "chunk_size_bytes": chunk_size,
        "sha256": original_sha256,
        "parts": [p.name for p in parts]
    }
    manifest_path = filepath.parent / f"{filepath.name}.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Split complete: {len(parts)} parts created. Manifest: {manifest_path.name}")
    return parts


def merge_file(target_file: Path, force: bool = False) -> bool:
    """Reassemble chunks into the original file if missing or if forced."""
    target_file = Path(target_file)
    manifest_path = target_file.parent / f"{target_file.name}.manifest.json"

    if not manifest_path.exists():
        # Fallback: look for parts matching pattern
        parts = sorted(target_file.parent.glob(f"{target_file.name}.part*"))
        if not parts:
            return False
        expected_sha256 = None
        expected_size = None
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        parts = [target_file.parent / p for p in manifest["parts"]]
        expected_sha256 = manifest.get("sha256")
        expected_size = manifest.get("total_size_bytes")

    if target_file.exists() and not force:
        # Verify existing file
        if expected_size is not None and target_file.stat().st_size == expected_size:
            print(f"Target file {target_file.name} already exists with expected size.")
            return True

    print(f"Reassembling {target_file.name} from {len(parts)} parts...")
    for part in parts:
        if not part.exists():
            raise FileNotFoundError(f"Missing part: {part}")

    temp_target = target_file.parent / f"{target_file.name}.assembling"
    with open(temp_target, "wb") as f_out:
        for part in parts:
            with open(part, "rb") as f_in:
                while chunk := f_in.read(4 * 1024 * 1024):
                    f_out.write(chunk)

    if expected_sha256:
        actual_sha256 = compute_sha256(temp_target)
        if actual_sha256 != expected_sha256:
            temp_target.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {target_file.name}!\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_sha256}"
            )
        print(f"Checksum verified: {actual_sha256[:16]}...")

    if target_file.exists():
        target_file.unlink()
    temp_target.rename(target_file)
    print(f"Successfully assembled {target_file.name} ({target_file.stat().st_size / (1024*1024):.2f} MB).")
    return True


def ensure_model_file(filepath: Path) -> Path:
    """Helper to ensure a model checkpoint exists, assembling from parts if necessary."""
    filepath = Path(filepath)
    if filepath.exists():
        return filepath
    manifest_path = filepath.parent / f"{filepath.name}.manifest.json"
    part_files = list(filepath.parent.glob(f"{filepath.name}.part*"))
    if manifest_path.exists() or part_files:
        merge_file(filepath)
    return filepath


def split_all_large_models(models_dir: Path = MODELS_DIR):
    """Scan and split all model files larger than 50MB."""
    print(f"Scanning for checkpoints > 50MB in {models_dir}...")
    targets = [
        models_dir / "best_model_tb.pth",
        models_dir / "densenet201" / "best_model.pth",
        models_dir / "fracture_convnext_base" / "best_model.pth",
    ]
    for target in targets:
        if target.exists():
            split_file(target)
        else:
            print(f"Target not found on disk: {target}")


def merge_all_models(models_dir: Path = MODELS_DIR, force: bool = False):
    """Scan and assemble all chunked models."""
    manifests = list(models_dir.rglob("*.manifest.json"))
    print(f"Found {len(manifests)} model manifests.")
    for m in manifests:
        original_name = m.name.replace(".manifest.json", "")
        target = m.parent / original_name
        merge_file(target, force=force)


def print_status(models_dir: Path = MODELS_DIR):
    """Print status of all model checkpoints and chunks."""
    print("=" * 60)
    print(f"MedGuard AI Model Weights Status ({models_dir})")
    print("=" * 60)
    checkpoints = [
        models_dir / "best_model.pth",
        models_dir / "best_model_tb.pth",
        models_dir / "densenet201" / "best_model.pth",
        models_dir / "fracture_convnext_base" / "best_model.pth",
    ]
    for ckpt in checkpoints:
        rel = ckpt.relative_to(models_dir)
        exists = ckpt.exists()
        size_mb = ckpt.stat().st_size / (1024*1024) if exists else 0
        parts = list(ckpt.parent.glob(f"{ckpt.name}.part*"))
        manifest = ckpt.parent / f"{ckpt.name}.manifest.json"
        print(f"Model: {rel}")
        print(f"  Assembled (.pth): {'YES' if exists else 'NO'} ({size_mb:.2f} MB)")
        print(f"  Chunk parts:      {len(parts)} parts")
        print(f"  Manifest:         {'YES' if manifest.exists() else 'NO'}")
        print()


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "split":
        split_all_large_models()
    elif action == "merge":
        force_flag = "--force" in sys.argv
        merge_all_models(force=force_flag)
    elif action == "status":
        print_status()
    else:
        print(f"Usage: python split_merge.py [split|merge|status] [--force]")
