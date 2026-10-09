"""
ml/src/chestmnist_dataset.py -- Dataset loader and transform pipeline for ChestMNIST 224x224.
=============================================================================================
Multi-label classification dataset (14 binary targets per radiograph).
Supports official MedMNIST train (78,468), val (11,219), and test (22,433) partitions.
Adapts 1-channel grayscale radiographs into 3-channel RGB tensors with ImageNet normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Path bootstrap
try:
    from config import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD
except ModuleNotFoundError:
    from ..config import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD

try:
    from src.densenet_model import CHESTMNIST_14_LABELS, NUM_CHEST_CLASSES
except ModuleNotFoundError:
    from densenet_model import CHESTMNIST_14_LABELS, NUM_CHEST_CLASSES


class ChestMNIST224Dataset(Dataset):
    """
    PyTorch Dataset for ChestMNIST 224x224 multi-label chest radiographs.
    Adapts 1-channel grayscale to 3-channel tensors for pretrained backbones.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ):
        self.images = images
        self.labels = labels.astype(np.float32)  # (N, 14) float32 for BCEWithLogitsLoss
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_arr = self.images[idx]
        target = self.labels[idx]

        # Squeeze potential trailing/leading singleton channel dimension
        if img_arr.ndim == 3 and img_arr.shape[0] == 1:
            img_arr = img_arr[0]
        elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
            img_arr = img_arr[:, :, 0]

        # Adapt grayscale to 3-channel RGB image for pretrained ImageNet backbones
        pil_img = Image.fromarray(img_arr.astype(np.uint8)).convert("RGB")

        if self.transform is not None:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.functional.to_tensor(pil_img)

        return tensor, torch.from_numpy(target).float()


def get_chestmnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns clinically appropriate transformations:
    - Train: Subtle horizontal flip (p=0.5), subtle rotation (+/- 7 deg), ToTensor, ImageNet normalization.
    - Eval: Deterministic ToTensor and ImageNet normalization.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def load_chestmnist_splits(
    data_dir: Union[str, Path] = DATA_DIR,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads official ChestMNIST 224x224 splits from the local .npz file.
    Does NOT fabricate data. Raises FileNotFoundError if dataset is missing or corrupt.
    """
    npz_path = Path(data_dir) / "chestmnist_224.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"ChestMNIST dataset not found at {npz_path}. "
            f"Please run 'python ml/download_chestmnist.py' to download the official dataset."
        )

    try:
        data = np.load(npz_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to load {npz_path}: {exc}. "
            f"The file appears incomplete or corrupt. Run 'python ml/download_chestmnist.py' to re-download."
        ) from exc

    required_keys = ["train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Corrupt ChestMNIST archive: missing key '{k}' in {npz_path}")

    train_imgs, train_lbls = data["train_images"], data["train_labels"]
    val_imgs, val_lbls = data["val_images"], data["val_labels"]
    test_imgs, test_lbls = data["test_images"], data["test_labels"]

    print(
        f"[ChestMNIST] Loaded official splits -- "
        f"Train: {len(train_imgs):,}, Val: {len(val_imgs):,}, Test: {len(test_imgs):,} "
        f"(Total: {len(train_imgs) + len(val_imgs) + len(test_imgs):,})"
    )

    return {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls),
    }


def compute_multilabel_pos_weights(
    train_labels: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes per-class positive weights for BCEWithLogitsLoss:
    pos_weight[c] = (num_negatives_c) / max(1, num_positives_c)
    Balances loss gradients for highly imbalanced thoracic findings.
    """
    n_samples = len(train_labels)
    pos_counts = np.sum(train_labels, axis=0)  # (14,)
    neg_counts = n_samples - pos_counts

    weights = []
    print("\n[ChestMNIST] Multi-label Class Imbalance & Positive Weights:")
    for i, name in enumerate(CHESTMNIST_14_LABELS):
        pos = int(pos_counts[i])
        neg = int(neg_counts[i])
        w = float(neg / max(1, pos))
        weights.append(w)
        prev = (pos / max(1, n_samples)) * 100.0
        print(f"  [{i:02d}] {name:<15} Pos: {pos:5d} ({prev:5.2f}%), Neg: {neg:5d}, pos_weight: {w:6.2f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_chestmnist_dataloaders(
    data_dir: Union[str, Path] = DATA_DIR,
    batch_size: int = 4,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Builds official Train, Val, and Test DataLoaders and computes pos_weights.
    """
    splits = load_chestmnist_splits(data_dir=data_dir)
    train_trans, eval_trans = get_chestmnist_transforms()

    train_ds = ChestMNIST224Dataset(splits["train"][0], splits["train"][1], transform=train_trans)
    val_ds = ChestMNIST224Dataset(splits["val"][0], splits["val"][1], transform=eval_trans)
    test_ds = ChestMNIST224Dataset(splits["test"][0], splits["test"][1], transform=eval_trans)

    is_cuda = device is not None and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weights = compute_multilabel_pos_weights(splits["train"][1], device=device)

    return train_loader, val_loader, test_loader, pos_weights
