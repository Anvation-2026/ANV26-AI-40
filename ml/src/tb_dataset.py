"""
TBX11K Dataset Loader -- MedGuard AI TB Classification Pipeline
================================================================
Parses the official TBX11K annotation JSONs (COCO format) and exposes
PyTorch Dataset objects for train / val splits.

Label strategy (binary TB triage):
  Non-TB  -> 0  (health/, sick/, extra/ subdirectories)
  TB      -> 1  (tb/ subdirectory)

Image preprocessing:
  - Resize to IMAGE_SIZE x IMAGE_SIZE
  - ImageNet mean/std normalisation (same as pneumonia pipeline)
  - Training augmentations: horizontal flip, rotation, color jitter

Usage:
  from src.tb_dataset import TBX11KDataset, get_tb_dataloaders
  train_dl, val_dl, pos_weight = get_tb_dataloaders(batch_size=32)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Early path bootstrap ─────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_ML_ROOT_TB = _Path(__file__).resolve().parent.parent
for _p in (str(_ML_ROOT_TB), str(_ML_ROOT_TB / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from config import (
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
)

# ── Dataset paths ────────────────────────────────────────────────────────────
# The zip extracts as: ml/data/TBX11K/TBX11K/{imgs,annotations,lists,...}
TBX11K_DIR      = DATA_DIR / "TBX11K" / "TBX11K"
TBX11K_IMGS_DIR = TBX11K_DIR / "imgs"
TBX11K_ANN_DIR  = TBX11K_DIR / "annotations" / "json"

# Official JSON split filenames present in the extracted archive
SPLIT_FILES: Dict[str, str] = {
    "train":    "all_train.json",       # 6888 images (TB + non-TB from all sources)
    "val":      "all_val.json",         # 2088 images
    "trainval": "all_trainval.json",    # 8976 images
    "test":     "all_test.json",        # 3302 images (no ground truth released)
    # Official TBX11K-only splits (no extra):
    "tbx_train":    "TBX11K_train.json",
    "tbx_val":      "TBX11K_val.json",
    "tbx_trainval": "TBX11K_trainval.json",
}

# Category IDs that represent tuberculosis in the COCO annotations
TB_CATEGORY_NAMES = frozenset([
    "ActiveTuberculosis",
    "ObsoletePulmonaryTuberculosis",
    "PulmonaryTuberculosis",
    "Tuberculosis",
])

# Binary label class names
TB_CLASS_NAMES = ["Non-TB", "TB"]


# ── Label derivation ─────────────────────────────────────────────────────────

def _label_from_file_path(fname: str) -> int:
    """
    Derive binary label from the image file path within the archive.
    TBX11K organises images in subdirectories:
        tb/     -> 1 (TB)
        health/ -> 0 (Non-TB, healthy)
        sick/   -> 0 (Non-TB, sick & non-TB)
        extra/  -> 0 (Non-TB, extra images)
        test/   -> -1 (unknown, no ground truth)
    """
    parts = Path(fname).parts
    subdir = parts[0].lower() if parts else ""
    if subdir == "tb":
        return 1
    if subdir in ("health", "sick", "extra"):
        return 0
    # Fallback to stem prefix
    stem = Path(fname).stem.lower()
    if any(stem.startswith(p) for p in ("activetb", "latenttb", "activelatenttb")):
        return 1
    if any(stem.startswith(p) for p in ("n", "healthy", "sick")):
        return 0
    return -1   # test set or unknown


def _label_from_annotation(image_id: int, ann_by_image: Dict[int, List[dict]],
                            cat_tb_ids: frozenset) -> int:
    """Returns 1 if any bbox annotation is a TB category, else 0."""
    anns = ann_by_image.get(image_id, [])
    for ann in anns:
        if ann["category_id"] in cat_tb_ids:
            return 1
    return 0


# ── Dataset class ────────────────────────────────────────────────────────────

class TBX11KDataset(Dataset):
    """
    PyTorch Dataset for TBX11K chest X-ray binary classification (Non-TB vs TB).

    Args:
        split:     One of 'train', 'val', 'trainval', 'test',
                   'tbx_train', 'tbx_val', 'tbx_trainval'.
        transform: Optional torchvision transform. Defaults to standard ImageNet transform.
        imgs_dir:  Path to the imgs/ directory.
        ann_dir:   Path to the annotations/json/ directory.
        skip_unknown_labels: If True, skips images whose labels cannot be determined.
    """

    def __init__(
        self,
        split:                str  = "train",
        transform                  = None,
        imgs_dir:             Path = TBX11K_IMGS_DIR,
        ann_dir:              Path = TBX11K_ANN_DIR,
        skip_unknown_labels:  bool = True,
    ):
        self.split               = split
        self.imgs_dir            = Path(imgs_dir)
        self.ann_dir             = Path(ann_dir)
        self.skip_unknown_labels = skip_unknown_labels
        self.transform = transform if transform is not None else self._default_transform(split)

        self.samples: List[Tuple[Path, int]] = []
        self._load_split()

    def _default_transform(self, split: str) -> transforms.Compose:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        resize    = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        if split in ("train", "tbx_train"):
            return transforms.Compose([
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([resize, transforms.ToTensor(), normalize])

    def _load_split(self) -> None:
        ann_filename = SPLIT_FILES.get(self.split)
        if ann_filename is None:
            raise ValueError(f"Unknown split '{self.split}'. Choose from: {list(SPLIT_FILES)}")

        ann_path = self.ann_dir / ann_filename
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                f"Please extract TBX11K.zip to {TBX11K_DIR.parent}"
            )

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Build TB category id set
        cat_tb_ids: set = set()
        for cat in coco.get("categories", []):
            if (cat["name"] in TB_CATEGORY_NAMES or
                    cat.get("supercategory", "") in TB_CATEGORY_NAMES):
                cat_tb_ids.add(cat["id"])

        # Index annotations by image id
        ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Build (path, label) sample list
        missing = 0
        skipped_label = 0
        for img_info in coco["images"]:
            img_id = img_info["id"]
            fname  = img_info["file_name"]
            img_path = self.imgs_dir / fname

            # Derive label
            label = _label_from_file_path(fname)
            if label == -1:
                # Try annotation-based fallback
                label = _label_from_annotation(img_id, ann_by_image, frozenset(cat_tb_ids))

            if label == -1 and self.skip_unknown_labels:
                skipped_label += 1
                continue

            if not img_path.exists():
                missing += 1
                continue

            self.samples.append((img_path, label))

        if missing:
            print(f"[TBX11KDataset/{self.split}] {missing} image files not found on disk (skipped).")
        if skipped_label:
            print(f"[TBX11KDataset/{self.split}] {skipped_label} images with unknown labels skipped.")

        n_non_tb = sum(1 for _, lbl in self.samples if lbl == 0)
        n_tb     = sum(1 for _, lbl in self.samples if lbl == 1)
        print(
            f"[TBX11KDataset/{self.split}] Loaded {len(self.samples)} samples | "
            f"Non-TB={n_non_tb}  TB={n_tb}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label

    @property
    def labels(self) -> List[int]:
        """All integer labels in dataset order."""
        return [lbl for _, lbl in self.samples]

    def class_counts(self) -> Tuple[int, int]:
        """Returns (n_non_tb, n_tb)."""
        lbl_list = self.labels
        return lbl_list.count(0), lbl_list.count(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_tb_pos_weight(dataset: TBX11KDataset, device: torch.device) -> torch.Tensor:
    """
    Computes inverse-frequency positive (TB) weight for BCEWithLogitsLoss.
    pos_weight = n_negative / n_positive
    """
    n_non_tb, n_tb = dataset.class_counts()
    if n_tb == 0:
        raise ValueError("Training set contains zero TB samples -- check annotation paths!")
    pw = n_non_tb / n_tb
    print(f"[TBX11K] pos_weight = {pw:.3f} (Non-TB:{n_non_tb} / TB:{n_tb})")
    return torch.tensor([pw], dtype=torch.float32, device=device)


def get_tb_dataloaders(
    batch_size:  int           = 32,
    num_workers: int           = NUM_WORKERS,
    device:      Optional[torch.device] = None,
    imgs_dir:    Path          = TBX11K_IMGS_DIR,
    ann_dir:     Path          = TBX11K_ANN_DIR,
    use_tbx_splits: bool       = False,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Constructs train/val DataLoaders for TBX11K binary classification.

    Args:
        use_tbx_splits: If True, uses TBX11K-only splits (6600/1800).
                        If False, uses all_* splits including extra images (6888/2088).

    Returns:
        (train_loader, val_loader, pos_weight)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_split  = "tbx_train" if use_tbx_splits else "train"
    val_split = "tbx_val"   if use_tbx_splits else "val"

    train_ds = TBX11KDataset(split=tr_split,  imgs_dir=imgs_dir, ann_dir=ann_dir)
    val_ds   = TBX11KDataset(split=val_split, imgs_dir=imgs_dir, ann_dir=ann_dir)

    pos_weight = compute_tb_pos_weight(train_ds, device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader, pos_weight


def verify_dataset_exists() -> bool:
    """Sanity-checks that TBX11K images and key annotation files are present."""
    ok = True
    if not TBX11K_IMGS_DIR.exists():
        print(f"[MISSING] imgs dir: {TBX11K_IMGS_DIR}")
        ok = False
    else:
        exts = [".png", ".jpg", ".jpeg"]
        n = sum(len(list(TBX11K_IMGS_DIR.glob(f"**/*{e}"))) for e in exts)
        print(f"[OK]      {n} images under {TBX11K_IMGS_DIR}")

    required = ("all_train.json", "all_val.json")
    for fname in required:
        ap = TBX11K_ANN_DIR / fname
        if ap.exists():
            print(f"[OK]      {fname}  ({ap.stat().st_size // 1024} KB)")
        else:
            print(f"[MISSING] {ap}")
            ok = False

    return ok


if __name__ == "__main__":
    print("=== TBX11K Dataset Verification ===")
    if verify_dataset_exists():
        ds = TBX11KDataset(split="train")
        img, lbl = ds[0]
        print(f"Sample: shape={img.shape}, label={lbl} ({TB_CLASS_NAMES[lbl]})")
    else:
        print("\nRun: python ml/download_tbx11k.py  to download the dataset first.")
