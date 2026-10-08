import cv2
import numpy as np
import pytest

from config import DATASET_PATH, MODELS_DIR
from src.dataset import get_split
from src.quality import QualityGate, assess_quality


def test_quality_gate_clean_image():
    val_images, _ = get_split("val", DATASET_PATH)
    clean = val_images[0]
    res = assess_quality(clean)
    assert res["status"] in ["acceptable", "poor"]  # Most clean images pass


def test_quality_gate_degradations():
    val_images, _ = get_split("val", DATASET_PATH)
    base = val_images[0]

    # 1. Extreme blur
    blurry = cv2.GaussianBlur(base, (0, 0), 15.0)
    res_b = assess_quality(blurry)
    assert res_b["status"] == "poor"
    assert "blur" in res_b["reasons"]

    # 2. Too dark
    dark = (base * 0.05).astype(np.uint8)
    res_d = assess_quality(dark)
    assert res_d["status"] == "poor"
    assert "too_dark" in res_d["reasons"]

    # 3. Too bright
    bright = np.clip(base.astype(float) + 200, 0, 255).astype(np.uint8)
    res_br = assess_quality(bright)
    assert res_br["status"] == "poor"
    assert "too_bright" in res_br["reasons"]

    # 4. Severe noise
    noisy = np.clip(base.astype(float) + np.random.normal(0, 80, (224, 224)), 0, 255).astype(np.uint8)
    res_n = assess_quality(noisy)
    assert res_n["status"] == "poor"
    assert "noisy" in res_n["reasons"]
