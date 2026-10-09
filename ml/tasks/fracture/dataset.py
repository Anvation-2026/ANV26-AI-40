"""
Dataset & Radiograph DataLoader Module
MedGuard AI - Bone Fracture Analysis Subsystem
Genuine ConvNeXt-Base Multi-Region Radiograph Evaluation

Handles:
- 16-bit Graz Pediatric Wrist Radiographs (mode I;16) with robust percentile scaling
- 8-bit FracAtlas Multi-Region Radiographs (mode L/RGB)
- Standardized 3-channel RGB conversion at 224x224
- Medically safe augmentations (subtle rotation/translation, no destructive warping)
- ImageNet normalization for ConvNeXt-Base
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from ml.tasks.fracture.config import (
    MANIFEST_TRAIN,
    MANIFEST_VAL,
    MANIFEST_TEST,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS
)


def load_radiograph_tensor(image_path: str) -> torch.Tensor:
    """
    Loads an X-ray image from disk and converts it to a normalized 3-channel FloatTensor in [0, 1].
    Accurately preserves the full dynamic range of 16-bit radiographs.
    
    Returns:
        Tensor of shape (3, H, W) with values in [0.0, 1.0]
    """
    img = Image.open(image_path)
    
    if img.mode == "I;16" or img.mode == "I":
        # 16-bit or 32-bit grayscale (Graz Pediatric Wrist)
        arr = np.array(img, dtype=np.float32)
        p_low, p_high = np.percentile(arr, (0.5, 99.5))
        if p_high > p_low:
            arr = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            arr = arr / (65535.0 if arr.max() > 255.0 else 255.0)
        
        # Expand to 3 channels: (H, W) -> (3, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        
    elif img.mode == "L":
        # 8-bit single-channel grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        
    elif img.mode == "RGB":
        # 8-bit 3-channel RGB
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        
    else:
        # Convert any other mode (RGBA, P, etc.) to RGB first
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

    return tensor.to(dtype=torch.float32)


class FractureDataset(Dataset):
    """
    PyTorch Dataset for Multi-Region Bone Fracture Analysis.
    Reads from a verified manifest CSV file.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split_name: str = "train",
        image_size: Tuple[int, int] = IMAGE_SIZE,
        is_training: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        self.split_name = split_name
        self.image_size = image_size
        self.is_training = is_training

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        # Base resize transform
        self.resize_op = T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        
        # Clinical-grade training augmentations (gentle, anatomy-preserving)
        if self.is_training:
            self.spatial_aug = T.RandomAffine(
                degrees=(-7, 7),
                translate=(0.04, 0.04),
                scale=(0.96, 1.04),
                interpolation=T.InterpolationMode.BILINEAR
            )
            self.color_aug = T.ColorJitter(brightness=0.10, contrast=0.10)
        else:
            self.spatial_aug = None
            self.color_aug = None

        # ImageNet normalization
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        label = float(row["fracture_label"])

        # Load raw tensor in [0, 1]
        tensor = load_radiograph_tensor(img_path)

        # Apply resizing
        tensor = self.resize_op(tensor)

        # Apply training augmentations
        if self.is_training:
            if self.spatial_aug is not None:
                tensor = self.spatial_aug(tensor)
            if self.color_aug is not None:
                tensor = self.color_aug(tensor)

        # Apply ImageNet normalization
        tensor = self.normalize(tensor)

        return {
            "image": tensor,
            "label": torch.tensor([label], dtype=torch.float32),
            "dataset_source": str(row.get("dataset_source", "unknown")),
            "image_id": str(row.get("image_id", "")),
            "anatomical_region": str(row.get("anatomical_region", "unknown")),
            "patient_id": str(row.get("patient_id", "unknown")),
            "image_path": img_path
        }


def get_fracture_dataloaders(
    train_manifest: str | Path = MANIFEST_TRAIN,
    val_manifest: str | Path = MANIFEST_VAL,
    test_manifest: Optional[str | Path] = MANIFEST_TEST,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, validation, and optional test splits.
    """
    train_ds = FractureDataset(train_manifest, split_name="train", image_size=image_size, is_training=True)
    val_ds = FractureDataset(val_manifest, split_name="val", image_size=image_size, is_training=False)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }

    if test_manifest and Path(test_manifest).exists():
        test_ds = FractureDataset(test_manifest, split_name="test", image_size=image_size, is_training=False)
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return loaders
