import io
import numpy as np
from PIL import Image
import pytest
import torch

from src.preprocessing import InvalidImageError, preprocess_image, to_gray_uint8, to_tensor


def test_array_vs_png_bytes_equality():
    # Proves a numpy array and saved PNG produce identical tensors
    rng = np.random.RandomState(42)
    sample_arr = rng.randint(0, 256, (224, 224), dtype=np.uint8)

    buf = io.BytesIO()
    Image.fromarray(sample_arr).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    t_arr, g_arr = preprocess_image(sample_arr)
    t_bytes, g_bytes = preprocess_image(png_bytes)

    assert np.array_equal(g_arr, g_bytes)
    assert torch.allclose(t_arr, t_bytes, atol=1e-5)


def test_various_formats_and_modes():
    # 1. RGB
    rgb_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    t_rgb, g_rgb = preprocess_image(rgb_arr)
    assert t_rgb.shape == (1, 3, 224, 224)
    assert g_rgb.shape == (224, 224)

    # 2. RGBA
    rgba_arr = np.random.randint(0, 256, (150, 120, 4), dtype=np.uint8)
    t_rgba, g_rgba = preprocess_image(rgba_arr)
    assert t_rgba.shape == (1, 3, 224, 224)

    # 3. Non-square grayscale
    gray_rec = np.random.randint(0, 256, (300, 200), dtype=np.uint8)
    t_rec, g_rec = preprocess_image(gray_rec)
    assert t_rec.shape == (1, 3, 224, 224)


def test_invalid_and_corrupt_inputs():
    with pytest.raises(InvalidImageError):
        preprocess_image(b"not an image file")

    with pytest.raises(InvalidImageError):
        preprocess_image(np.zeros((0, 0), dtype=np.uint8))

    with pytest.raises(InvalidImageError):
        preprocess_image(np.zeros((1, 1), dtype=np.uint8))  # too tiny
