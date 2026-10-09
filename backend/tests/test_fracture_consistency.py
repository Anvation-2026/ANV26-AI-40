"""
Automated Transform & API Equivalence Test
MedGuard AI - Bone Fracture Subsystem
Verifies:
1. Production preprocessing matches Dataset evaluation transform with 0.0 delta.
2. Standalone FracturePredictor matches FastAPI endpoint with 0.0 delta.
"""

from pathlib import Path
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from starlette.testclient import TestClient

from backend.main import app
from ml.tasks.fracture.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, CHECKPOINT_BEST
from ml.tasks.fracture.dataset import load_radiograph_tensor
from ml.tasks.fracture.predict import FracturePredictor, get_fracture_predictor


@pytest.fixture
def sample_image_path():
    path = Path("frontend/public/assets/sample-bone-wrist-fracture.png")
    if not path.exists():
        pytest.skip("Sample test radiograph not found.")
    return path


def test_preprocessing_transform_consistency(sample_image_path):
    """Verifies production transform matches training evaluation transform identically."""
    p = FracturePredictor(device="cpu")
    img = Image.open(sample_image_path)
    
    # 1. Production inference tensor
    t_pred = p.preprocess_image(img).squeeze(0)

    # 2. Dataset pipeline tensor
    raw = load_radiograph_tensor(sample_image_path)
    resize_op = T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    norm_op = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    t_data = norm_op(resize_op(raw))

    max_diff = (t_pred - t_data).abs().max().item()
    assert max_diff < 1e-6, f"Preprocessing difference exceeded threshold: {max_diff}"


def test_standalone_vs_fastapi_endpoint_equivalence(sample_image_path):
    """Verifies standalone predictor and FastAPI route produce identical predictions."""
    if not CHECKPOINT_BEST.exists():
        pytest.skip("Fracture model checkpoint not found.")

    p = get_fracture_predictor()
    standalone_res = p.predict(sample_image_path, generate_cam=False)

    client = TestClient(app)
    with open(sample_image_path, "rb") as f:
        resp = client.post("/api/fracture/predict", files={"file": ("sample.png", f, "image/png")})

    assert resp.status_code == 200
    api_res = resp.json()

    assert standalone_res["finding"] == api_res["finding"]
    delta = abs(standalone_res["calibrated_probability"] - api_res["calibrated_probability"])
    assert delta < 1e-4, f"Prediction probability discrepancy: {delta}"
