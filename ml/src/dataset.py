import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import urllib.request

# Path bootstrap
from src import _ML_ROOT  # noqa: F401 - side effect: adds ml/ to sys.path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    CLASS_NAMES,
    DATA_DIR,
    DATASET_DOWNLOAD_URL,
    DATASET_PATH,
    DEFAULT_BATCH_SIZE,
    EXPECTED_SPLIT_SIZES,
    IMAGE_SIZE,
    NUM_WORKERS,
    POSITIVE_CLASS,
    POSITIVE_CLASS_IDX,
    REPORTS_DIR,
    SEED,
)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def ensure_dataset_downloaded(
    dest_path: Path = DATASET_PATH,
    url: str = DATASET_DOWNLOAD_URL,
) -> Path:
    """Ensures that the PneumoniaMNIST 224 npz file exists, downloading if necessary."""
    dest_path = Path(dest_path)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Dataset] Downloading PneumoniaMNIST 224 from {url}...")
    temp_path = dest_path.with_suffix(".tmp")

    def report_progress(blocks_transferred, block_size, total_size):
        downloaded = blocks_transferred * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(
                f"\r[Dataset] Download progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)"
            )
        else:
            sys.stdout.write(f"\r[Dataset] Downloaded: {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)
    print()
    temp_path.replace(dest_path)
    print(f"[Dataset] Download completed -> {dest_path}")
    return dest_path


def load_npz(path: Path = DATASET_PATH) -> Dict[str, np.ndarray]:
    """Loads the npz archive and returns arrays dictionary."""
    path = Path(path)
    if not path.exists():
        ensure_dataset_downloaded(path)
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_split(name: str, path: Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieves images and labels for a split ('train', 'val', or 'test')."""
    data = load_npz(path)
    img_key = f"{name}_images"
    lbl_key = f"{name}_labels"
    if img_key not in data or lbl_key not in data:
        raise KeyError(f"Split {name} not found in npz. Available keys: {list(data.keys())}")
    return data[img_key], data[lbl_key]


class PneumoniaDataset(Dataset):
    """
    PyTorch Dataset for PneumoniaMNIST 224.
    Images are (N, 224, 224) uint8 grayscale.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        self.images = images
        # Labels are shaped (N, 1) in MedMNIST, flatten to 1D
        self.labels = labels.squeeze()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_arr = self.images[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            tensor = self.transform(img_arr)
        else:
            # Fallback simple tensor conversion if no transform provided
            tensor = torch.from_numpy(img_arr).float() / 255.0
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

        target = torch.tensor(label, dtype=torch.long)
        return tensor, target


def get_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    augment: bool = True,
    path: Path = DATASET_PATH,
    train_transform=None,
    eval_transform=None,
) -> Dict[str, DataLoader]:
    """Creates train, val, and test DataLoaders."""
    # Import preprocessing transforms if not explicitly passed
    if train_transform is None or eval_transform is None:
        from src.preprocessing import get_eval_transform, get_train_transform
        train_transform = get_train_transform() if augment else get_eval_transform()
        eval_transform = get_eval_transform()


    train_imgs, train_lbls = get_split("train", path)
    val_imgs, val_lbls = get_split("val", path)
    test_imgs, test_lbls = get_split("test", path)

    train_ds = PneumoniaDataset(train_imgs, train_lbls, transform=train_transform)
    val_ds = PneumoniaDataset(val_imgs, val_lbls, transform=eval_transform)
    test_ds = PneumoniaDataset(test_imgs, test_lbls, transform=eval_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def dataset_summary(path: Path = DATASET_PATH) -> Dict:
    """
    Computes and verifies dataset characteristics, class distributions,
    and exact-duplicate image overlaps across splits.
    """
    path = ensure_dataset_downloaded(path)
    file_sha256 = compute_file_sha256(path)
    file_size_bytes = path.stat().st_size

    data = load_npz(path)
    keys_info = {}
    split_counts = {}
    class_counts_by_split = {}
    hashes_by_split = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        imgs = data[f"{split}_images"]
        lbls = data[f"{split}_labels"]
        split_counts[split] = len(imgs)

        # Record keys info
        keys_info[f"{split}_images"] = {
            "shape": list(imgs.shape),
            "dtype": str(imgs.dtype),
            "min": int(imgs.min()),
            "max": int(imgs.max()),
        }
        keys_info[f"{split}_labels"] = {
            "shape": list(lbls.shape),
            "dtype": str(lbls.dtype),
            "unique_values": sorted([int(x) for x in np.unique(lbls)]),
        }

        # Class counts
        flat_lbls = lbls.squeeze()
        counts = {
            "normal (0)": int(np.sum(flat_lbls == 0)),
            "pneumonia (1)": int(np.sum(flat_lbls == 1)),
        }
        class_counts_by_split[split] = counts

        # Compute image hashes for duplicate detection
        split_hashes = [hashlib.sha256(img.tobytes()).hexdigest() for img in imgs]
        hashes_by_split[split] = split_hashes

    # Exact duplicate analysis
    train_set = set(hashes_by_split["train"])
    val_set = set(hashes_by_split["val"])
    test_set = set(hashes_by_split["test"])

    overlap_train_val = len(train_set.intersection(val_set))
    overlap_train_test = len(train_set.intersection(test_set))
    overlap_val_test = len(val_set.intersection(test_set))

    within_split_dups = {
        split: len(hashes_by_split[split]) - len(set(hashes_by_split[split]))
        for split in ["train", "val", "test"]
    }

    # Verify split size expectations
    split_matches = {
        split: (split_counts[split] == EXPECTED_SPLIT_SIZES.get(split))
        for split in ["train", "val", "test"]
    }
    all_splits_match = all(split_matches.values())

    summary = {
        "dataset_name": "PneumoniaMNIST+ (224x224)",
        "file_path": str(path),
        "file_sha256": file_sha256,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "keys": keys_info,
        "split_sizes": split_counts,
        "expected_sizes": EXPECTED_SPLIT_SIZES,
        "split_sizes_match_expected": all_splits_match,
        "class_counts": class_counts_by_split,
        "label_mapping": {0: "normal", 1: "pneumonia"},
        "positive_class": POSITIVE_CLASS,
        "exact_duplicates": {
            "within_split": within_split_dups,
            "train_val_overlap": overlap_train_val,
            "train_test_overlap": overlap_train_test,
            "val_test_overlap": overlap_val_test,
        },
        "notes": "Grayscale uint8 pediatric chest X-rays. Official benchmark.",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PneumoniaMNIST Dataset Check")
    parser.add_argument("--check", action="store_true", help="Run full dataset check and generate summary")
    parser.add_argument("--data-path", type=str, default=str(DATASET_PATH), help="Path to npz file")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    summary = dataset_summary(data_path)

    print("\n" + "=" * 50)
    print("      DATASET VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"File: {summary['file_path']}")
    print(f"SHA-256: {summary['file_sha256']}")
    print(f"Size: {summary['file_size_mb']} MB")
    print("\nSplit Sizes (Actual vs Expected):")
    for s, count in summary["split_sizes"].items():
        exp = summary["expected_sizes"].get(s)
        match_str = "MATCH" if count == exp else f"MISMATCH (Expected {exp})"
        print(f"  - {s.upper():5s}: {count:5d}  [{match_str}]")
    print(f"Total: {sum(summary['split_sizes'].values())} samples")

    print("\nClass Distributions:")
    for s, c in summary["class_counts"].items():
        total = sum(c.values())
        p_pct = (c['pneumonia (1)'] / total) * 100
        n_pct = (c['normal (0)'] / total) * 100
        print(f"  - {s.upper():5s}: Normal={c['normal (0)']} ({n_pct:.1f}%), Pneumonia={c['pneumonia (1)']} ({p_pct:.1f}%)")

    print("\nExact Duplicate Analysis:")
    for s, count in summary["exact_duplicates"]["within_split"].items():
        print(f"  - Within {s}: {count} duplicates")
    print(f"  - Train vs Val overlap: {summary['exact_duplicates']['train_val_overlap']}")
    print(f"  - Train vs Test overlap: {summary['exact_duplicates']['train_test_overlap']}")
    print(f"  - Val vs Test overlap: {summary['exact_duplicates']['val_test_overlap']}")
    print("=" * 50)
    print(f"Summary report written to {REPORTS_DIR / 'dataset_summary.json'}\n")
