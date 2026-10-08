import numpy as np
import pytest

from config import DATASET_PATH, MODELS_DIR
from src.dataset import get_split
from src.ood import OODDetector, check_modality_color


def test_modality_check_color_vs_gray():
    # 1. Grayscale array
    gray = np.full((100, 100), 120, dtype=np.uint8)
    assert not check_modality_color(gray)

    # 2. RGB image that is strictly grayscale (R==G==B)
    gray_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    assert not check_modality_color(gray_rgb)

    # 3. True color image (strong color saturation)
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    color[:, :, 0] = 200
    color[:, :, 1] = 50
    assert check_modality_color(color)


def test_ood_detector_with_stats():
    ood_stats_path = MODELS_DIR / "ood_stats.npz"
    if not ood_stats_path.exists():
        pytest.skip("ood_stats.npz not yet generated")

    detector = OODDetector(ood_stats_path)

    # Random noise feature check
    fake_noise_feats = np.random.randn(512) * 50.0
    score = detector.mahalanobis_score(fake_noise_feats)
    assert score > detector.ood_threshold
