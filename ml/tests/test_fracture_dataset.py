"""
Unit Tests for Fracture Dataset & DataLoaders
MedGuard AI - Bone Fracture Analysis Subsystem
"""

import os
from pathlib import Path
import pytest
import torch
import pandas as pd

from ml.tasks.fracture.config import (
    MANIFEST_TRAIN,
    MANIFEST_VAL,
    MANIFEST_TEST,
    MANIFEST_QUARANTINE
)
from ml.tasks.fracture.dataset import (
    FractureDataset,
    get_fracture_dataloaders,
    load_radiograph_tensor
)


def test_manifest_existence_and_schemas():
    """Verify all manifests exist and contain required columns."""
    for p in [MANIFEST_TRAIN, MANIFEST_VAL, MANIFEST_TEST]:
        assert p.exists(), f"Manifest {p} does not exist"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Manifest {p} is empty"
        required_cols = [
            "dataset_source", "image_path", "image_id", "patient_id",
            "study_id", "fracture_label", "anatomical_region", "split"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column {col} in {p}"


def test_no_image_leakage_between_splits():
    """Verify zero overlap of image paths across train, val, and test splits."""
    df_train = pd.read_csv(MANIFEST_TRAIN)
    df_val = pd.read_csv(MANIFEST_VAL)
    df_test = pd.read_csv(MANIFEST_TEST)

    train_paths = set(df_train["image_path"])
    val_paths = set(df_val["image_path"])
    test_paths = set(df_test["image_path"])

    assert len(train_paths.intersection(val_paths)) == 0, "Image leakage between Train and Val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Image leakage between Train and Test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Image leakage between Val and Test!"


def test_no_patient_leakage_in_graz():
    """Verify zero Graz patient leakage across train, val, and test splits."""
    df_train = pd.read_csv(MANIFEST_TRAIN)
    df_val = pd.read_csv(MANIFEST_VAL)
    df_test = pd.read_csv(MANIFEST_TEST)

    graz_tr_pts = set(df_train[df_train["dataset_source"] == "graz_pediatric_wrist"]["patient_id"])
    graz_val_pts = set(df_val[df_val["dataset_source"] == "graz_pediatric_wrist"]["patient_id"])
    graz_te_pts = set(df_test[df_test["dataset_source"] == "graz_pediatric_wrist"]["patient_id"])

    assert len(graz_tr_pts.intersection(graz_val_pts)) == 0, "Graz patient leakage between Train and Val!"
    assert len(graz_tr_pts.intersection(graz_te_pts)) == 0, "Graz patient leakage between Train and Test!"
    assert len(graz_val_pts.intersection(graz_te_pts)) == 0, "Graz patient leakage between Val and Test!"


def test_dataset_item_shapes_and_types():
    """Verify dataset __getitem__ output shapes and data types for both datasets."""
    ds = FractureDataset(MANIFEST_TRAIN, split_name="train", is_training=False)
    assert len(ds) > 0
    item = ds[0]

    assert "image" in item
    assert "label" in item
    assert "dataset_source" in item

    img_t = item["image"]
    label_t = item["label"]

    assert isinstance(img_t, torch.Tensor)
    assert img_t.shape == (3, 224, 224)
    assert img_t.dtype == torch.float32

    assert isinstance(label_t, torch.Tensor)
    assert label_t.shape == (1,)
    assert label_t.item() in [0.0, 1.0]


def test_dataloader_batch_collation():
    """Verify DataLoader collation and shapes."""
    loaders = get_fracture_dataloaders(
        train_manifest=MANIFEST_TRAIN,
        val_manifest=MANIFEST_VAL,
        batch_size=4,
        num_workers=0
    )
    batch = next(iter(loaders["train"]))

    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["label"].shape == (4, 1)
