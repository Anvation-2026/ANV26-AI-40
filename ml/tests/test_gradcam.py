import numpy as np
import pytest
import torch
import torchvision.models as models

from config import MODELS_DIR
from src.gradcam import GradCAM
from src.model import build_model, load_checkpoint


def test_gradcam_computation_and_range():
    ckpt_path = MODELS_DIR / "best_model.pth"
    if not ckpt_path.exists():
        pytest.skip("best_model.pth missing")

    model, _ = load_checkpoint(ckpt_path, device="cpu")
    gradcam = GradCAM(model, target_layer_name="layer4")

    dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    cam = gradcam.compute_cam(dummy_tensor, target_class=1)

    assert cam.shape == (224, 224)
    assert not np.isnan(cam).any()
    assert 0.0 <= cam.min() <= cam.max() <= 1.0


def test_gradcam_saliency_sanity_randomized_model():
    ckpt_path = MODELS_DIR / "best_model.pth"
    if not ckpt_path.exists():
        pytest.skip("best_model.pth missing")

    trained_model, _ = load_checkpoint(ckpt_path, device="cpu")
    random_model = build_model(pretrained=False)

    cam_engine_trained = GradCAM(trained_model)
    cam_engine_random = GradCAM(random_model)

    torch.manual_seed(42)
    sample_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    cam_trained = cam_engine_trained.compute_cam(sample_tensor, target_class=1)
    cam_random = cam_engine_random.compute_cam(sample_tensor, target_class=1)

    # Trained model saliency map should differ noticeably from randomized model map
    diff = np.mean(np.abs(cam_trained - cam_random))
    assert diff > 0.01
