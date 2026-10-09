"""
Download and Verify ChestMNIST 224x224 Dataset -- MedGuard AI
=============================================================
Downloads the official MedMNIST v2 ChestMNIST 224x224 dataset from Zenodo,
with automatic resume on connection drop, chunk verification, and MD5 integrity check.

Official Details:
  - Source: Zenodo (Record 10519652)
  - URL: https://zenodo.org/records/10519652/files/chestmnist_224.npz?download=1
  - Size: 3,889,293,042 bytes (~3.89 GB)
  - MD5: 45bd33e6f06c3e8cdb481c74a89152aa
  - Splits: Train (78,468), Val (11,219), Test (22,433)
"""

import hashlib
import os
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np

ZENODO_URL = "https://zenodo.org/records/10519652/files/chestmnist_224.npz?download=1"
EXPECTED_MD5 = "45bd33e6f06c3e8cdb481c74a89152aa"
EXPECTED_SIZE = 3889293042

ML_ROOT = Path(__file__).resolve().parent
DATA_DIR = ML_ROOT / "data"
TARGET_FILE = DATA_DIR / "chestmnist_224.npz"
TEMP_FILE = DATA_DIR / "chestmnist_224.npz.part"


def verify_file(filepath: Path) -> bool:
    """Verifies that the file exists, has correct size, and can be read by numpy."""
    if not filepath.exists():
        return False
    size = filepath.stat().st_size
    print(f"[Verify] File size: {size:,} bytes (expected {EXPECTED_SIZE:,} bytes)")
    if size != EXPECTED_SIZE:
        print(f"[Verify] Size mismatch! Got {size}, expected {EXPECTED_SIZE}")
        return False

    print("[Verify] Checking numpy readability and array shapes...")
    try:
        data = np.load(filepath)
        keys = list(data.keys())
        print(f"[Verify] Keys present: {keys}")
        for k in ["train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"]:
            if k not in data:
                print(f"[Verify] Missing key: {k}")
                return False
            arr = data[k]
            print(f"  - {k:<15}: shape={arr.shape}, dtype={arr.dtype}")

        assert len(data["train_images"]) == 78468, "Train count mismatch"
        assert len(data["val_images"]) == 11219, "Val count mismatch"
        assert len(data["test_images"]) == 22433, "Test count mismatch"
        print("[Verify] Dataset arrays and split counts verified successfully!")
        return True
    except Exception as exc:
        print(f"[Verify] Error reading file: {exc}")
        return False


def compute_md5(filepath: Path) -> str:
    """Computes MD5 hash in chunks."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):  # 8MB chunks
            h.update(chunk)
    return h.hexdigest()


def download_dataset() -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check if existing target file is already complete and valid
    if TARGET_FILE.exists():
        print(f"[Check] Checking existing {TARGET_FILE.name}...")
        if verify_file(TARGET_FILE):
            print("[OK] Dataset is already fully downloaded and valid!")
            return True
        else:
            print("[Warning] Existing file is incomplete or corrupted. Removing bad file...")
            TARGET_FILE.unlink(missing_ok=True)

    print("\n" + "=" * 65)
    print("  DOWNLOADING CHESTMNIST 224x224 FROM OFFICIAL ZENODO REPO")
    print("=" * 65)
    print(f"URL: {ZENODO_URL}")
    print(f"Target: {TARGET_FILE}")
    print(f"Expected Size: {EXPECTED_SIZE / (1024**3):.2f} GB ({EXPECTED_SIZE:,} bytes)\n")

    max_retries = 50
    retries = 0

    while retries < max_retries:
        current_bytes = TEMP_FILE.stat().st_size if TEMP_FILE.exists() else 0
        if current_bytes >= EXPECTED_SIZE:
            print("[Download] Reached expected byte count. Checking integrity...")
            break

        headers = {
            "User-Agent": "MedGuardAI-Downloader/1.1",
        }
        if current_bytes > 0:
            headers["Range"] = f"bytes={current_bytes}-{EXPECTED_SIZE - 1}"
            print(f"[Resume] Resuming from byte {current_bytes:,} ({current_bytes / (1024**3):.2f} GB)...")

        req = urllib.request.Request(ZENODO_URL, headers=headers)
        start_time = time.time()
        last_print = 0.0
        bytes_in_session = 0

        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(TEMP_FILE, "ab") as out_f:
                chunk_size = 1024 * 1024 * 4  # 4MB chunks
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    bytes_in_session += len(chunk)
                    total_downloaded = current_bytes + bytes_in_session

                    now = time.time()
                    if now - last_print > 4.0 or total_downloaded >= EXPECTED_SIZE:
                        last_print = now
                        elapsed = now - start_time
                        speed_mb = (bytes_in_session / (1024 * 1024)) / max(1e-3, elapsed)
                        percent = (total_downloaded / EXPECTED_SIZE) * 100.0
                        remaining_bytes = EXPECTED_SIZE - total_downloaded
                        eta_sec = remaining_bytes / max(1.0, (bytes_in_session / max(1e-3, elapsed)))
                        eta_min = eta_sec / 60.0
                        print(
                            f"Progress: {total_downloaded / (1024**3):.2f}/{EXPECTED_SIZE / (1024**3):.2f} GB "
                            f"({percent:.1f}%) | Speed: {speed_mb:.1f} MB/s | ETA: {eta_min:.1f} min",
                            flush=True,
                        )

            current_bytes = TEMP_FILE.stat().st_size if TEMP_FILE.exists() else 0
            if current_bytes >= EXPECTED_SIZE:
                break
            else:
                print(f"[Info] Connection dropped at {current_bytes / (1024**3):.2f} GB. Auto-resuming in 2s...")
                time.sleep(2)
                retries += 1

        except Exception as exc:
            print(f"[Warning] Network error: {exc}. Retrying in 3s...")
            time.sleep(3)
            retries += 1

    final_size = TEMP_FILE.stat().st_size if TEMP_FILE.exists() else 0
    if final_size != EXPECTED_SIZE:
        print(f"[ERROR] Download finished but size is {final_size} != {EXPECTED_SIZE}")
        return False

    print("\n[Download] Complete! Verifying MD5 checksum...")
    file_md5 = compute_md5(TEMP_FILE)
    print(f"[MD5] Computed: {file_md5} | Expected: {EXPECTED_MD5}")
    if file_md5.lower() != EXPECTED_MD5.lower():
        print("[ERROR] MD5 checksum mismatch! File is corrupted.")
        return False

    # Atomic rename
    if TARGET_FILE.exists():
        TARGET_FILE.unlink()
    TEMP_FILE.rename(TARGET_FILE)
    print(f"\n[OK] Successfully downloaded and verified: {TARGET_FILE}")
    return verify_file(TARGET_FILE)


if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
