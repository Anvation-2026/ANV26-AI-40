import numpy as np
import pytest
import torch

from config import DATASET_PATH
from src.dataset import get_dataloaders, get_split, load_npz


def test_dataset_splits_and_shapes():
    data = load_npz(DATASET_PATH)
    assert "train_images" in data
    assert "val_images" in data
    assert "test_images" in data

    train_imgs, train_lbls = get_split("train", DATASET_PATH)
    val_imgs, val_lbls = get_split("val", DATASET_PATH)
    test_imgs, test_lbls = get_split("test", DATASET_PATH)

    assert len(train_imgs) == 4708
    assert len(val_imgs) == 524
    assert len(test_imgs) == 624

    assert train_imgs.shape == (4708, 224, 224)
    assert train_imgs.dtype == np.uint8
    assert set(np.unique(train_lbls.squeeze())).issubset({0, 1})
    assert set(np.unique(val_lbls.squeeze())).issubset({0, 1})
    assert set(np.unique(test_lbls.squeeze())).issubset({0, 1})


def test_dataloaders():
    loaders = get_dataloaders(batch_size=8, augment=False)
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    batch_x, batch_y = next(iter(loaders["val"]))
    assert batch_x.shape == (8, 3, 224, 224)
    assert batch_x.dtype == torch.float32
    assert batch_y.shape == (8,)
