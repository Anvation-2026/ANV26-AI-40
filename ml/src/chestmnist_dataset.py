"""
ml/src/chestmnist_dataset.py — Dataset loader and transform pipeline for ChestMNIST 224x224.

Multi-label classification dataset (14 binary targets per radiograph).
Supports official MedMNIST train, val, and test partitions.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD

try:
    from src.densenet_model import CHESTMNIST_14_LABELS, NUM_CHEST_CLASSES
except ModuleNotFoundError:
    from densenet_model import CHESTMNIST_14_LABELS, NUM_CHEST_CLASSES


class ChestMNIST224Dataset(Dataset):
    """
    PyTorch Dataset for ChestMNIST 224x224 multi-label chest radiographs.
    Converts 1-channel grayscale to 3-channel tensors for pretrained backbones.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ):
        self.images = images
        self.labels = labels.astype(np.float32)  # (N, 14) float for BCEWithLogitsLoss
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_arr = self.images[idx]
        target = self.labels[idx]

        # Ensure 2D uint8
        if img_arr.ndim == 3 and img_arr.shape[0] == 1:
            img_arr = img_arr[0]
        elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
            img_arr = img_arr[:, :, 0]

        # Convert to PIL RGB for torchvision transforms
        pil_img = Image.fromarray(img_arr.astype(np.uint8)).convert("RGB")

        if self.transform is not None:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.functional.to_tensor(pil_img)

        return tensor, torch.from_numpy(target).float()


def get_chestmnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns medically appropriate augmentations:
    - Train: slight rotation (+/- 7 deg), subtle horizontal flip (p=0.5), slight scale (0.95-1.05), ImageNet normalization.
    - Eval: deterministic resize, to_tensor, ImageNet normalization.
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


def load_or_download_chestmnist(
    data_dir: Path,
    download_if_missing: bool = True,
    smoke_fallback: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads official ChestMNIST 224x224 splits from local .npz file,
    downloads using medmnist if missing, or uses local radiographs for smoke profiling.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "chestmnist_224.npz"

    if not npz_path.exists():
        if smoke_fallback:
            print(f"[ChestMNIST] Full 3.7GB chestmnist_224.npz not yet downloaded. Using local radiograph split for smoke/profiling.")
            pneu_path = data_dir / "pneumoniamnist_224.npz"
            if pneu_path.exists():
                pdata = np.load(pneu_path)
                t_imgs = pdata["train_images"][:256]
                v_imgs = pdata["val_images"][:128]
                te_imgs = pdata["test_images"][:128]
                rng = np.random.RandomState(42)
                t_lbls = (rng.rand(len(t_imgs), NUM_CHEST_CLASSES) > 0.85).astype(np.float32)
                v_lbls = (rng.rand(len(v_imgs), NUM_CHEST_CLASSES) > 0.85).astype(np.float32)
                te_lbls = (rng.rand(len(te_imgs), NUM_CHEST_CLASSES) > 0.85).astype(np.float32)
                return {
                    "train": (t_imgs, t_lbls),
                    "val": (v_imgs, v_lbls),
                    "test": (te_imgs, te_lbls),
                }

        if download_if_missing:
            print(f"[ChestMNIST] Downloading chestmnist_224.npz to {data_dir} via MedMNIST (3.7 GB)...")
            import medmnist
            from medmnist import ChestMNIST
            ChestMNIST(split="train", download=True, size=224, root=str(data_dir))
        else:
            raise FileNotFoundError(f"ChestMNIST dataset not found at {npz_path}")

    print(f"[ChestMNIST] Loading dataset splits from {npz_path}...")
    npz_data = np.load(npz_path)

    train_imgs = npz_data["train_images"]
    train_lbls = npz_data["train_labels"]
    val_imgs = npz_data["val_images"]
    val_lbls = npz_data["val_labels"]
    test_imgs = npz_data["test_images"]
    test_lbls = npz_data["test_labels"]

    print(f"[ChestMNIST] Split sizes — Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    return {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls),
    }


def compute_multilabel_pos_weights(train_labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Computes per-class positive weights for BCEWithLogitsLoss:
    pos_weight[c] = (num_negatives_c) / max(1, num_positives_c)
    """
    n_samples = len(train_labels)
    pos_counts = np.sum(train_labels, axis=0)  # (14,)
    neg_counts = n_samples - pos_counts

    weights = []
    print("\n[ChestMNIST] Class Imbalance and Positive Weights:")
    for i, name in enumerate(CHESTMNIST_14_LABELS):
        pos = int(pos_counts[i])
        neg = int(neg_counts[i])
        w = float(neg / max(1, pos))
        weights.append(w)
        prev = (pos / n_samples) * 100.0
        print(f"  [{i:02d}] {name:<15} Pos: {pos:5d} ({prev:5.2f}%), Neg: {neg:5d}, pos_weight: {w:.3f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)
