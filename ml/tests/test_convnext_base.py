"""
Architecture and Smoke Tests for ConvNeXt-Base Fracture Subsystem
MedGuard AI - Bone Fracture Analysis Subsystem
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ml.tasks.fracture.model import ConvNeXtFractureClassifier, create_fracture_model
from ml.tasks.fracture.gradcam import ConvNeXtGradCAM


def test_convnext_architecture_instantiation():
    """Verify ConvNeXt-Base backbone instantiation and parameter count."""
    model = ConvNeXtFractureClassifier(pretrained=False)
    summary = model.get_parameter_summary()

    # Base convnext_base has ~88.59M params + 3-stage MLP head ~0.65M params = ~89.2M params
    assert summary["total_params"] > 88_000_000, f"Expected >88M params, got {summary['total_params']:,}"
    assert model.feature_dim == 1024


def test_forward_pass_and_shapes():
    """Verify forward pass output shape is (B, 1) and feature extraction is (B, 1024)."""
    model = ConvNeXtFractureClassifier(pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model.extract_features(dummy_input)
        logits = model(dummy_input)
        probs = model.predict_probability(dummy_input)

    assert feats.shape == (2, 1024), f"Unexpected feature shape: {feats.shape}"
    assert logits.shape == (2, 1), f"Unexpected logit shape: {logits.shape}"
    assert probs.shape == (2, 1), f"Unexpected prob shape: {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]"


def test_freeze_and_unfreeze_controls():
    """Verify backbone freezing restricts trainable parameters to the MLP head only."""
    model = ConvNeXtFractureClassifier(pretrained=False)
    
    # Freeze backbone
    model.freeze_backbone()
    frozen_summary = model.get_parameter_summary()
    assert frozen_summary["trainable_params"] < 1_000_000, "Frozen model should have <1M trainable parameters"
    assert frozen_summary["frozen_params"] > 87_000_000, "Frozen model should have >87M frozen parameters"

    # Unfreeze selective stage
    model.unfreeze_stage(stage_idx=7)
    stage7_summary = model.get_parameter_summary()
    assert stage7_summary["trainable_params"] > frozen_summary["trainable_params"]

    # Unfreeze entire network
    model.unfreeze_backbone()
    unfrozen_summary = model.get_parameter_summary()
    assert unfrozen_summary["frozen_params"] == 0


def test_cuda_forward_backward_pass():
    """Verify CUDA forward and backward pass with BCEWithLogitsLoss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtFractureClassifier(pretrained=False).to(device)
    model.freeze_backbone()

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    dummy_x = torch.randn(2, 3, 224, 224, device=device)
    dummy_y = torch.tensor([[1.0], [0.0]], device=device)

    # Forward
    logits = model(dummy_x)
    loss = criterion(logits, dummy_y)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check gradients exist on classifier
    for param in model.classifier.parameters():
        assert param.grad is not None


def test_checkpoint_save_and_reload():
    """Verify checkpoint persistence and exact state reloading."""
    model1 = ConvNeXtFractureClassifier(pretrained=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pth"
        model1.save_checkpoint(ckpt_path, epoch=1, val_metric=0.85)

        assert ckpt_path.exists()
        model2 = ConvNeXtFractureClassifier(pretrained=False)
        info = model2.load_checkpoint(ckpt_path)

        assert info["epoch"] == 1
        assert info["best_val_metric"] == 0.85

        # Verify weights match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


def test_gradcam_generation():
    """Verify Grad-CAM attribution generation and hook mechanics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtFractureClassifier(pretrained=False).to(device)
    cam = ConvNeXtGradCAM(model)

    dummy_x = torch.randn(1, 3, 224, 224, device=device)
    heatmap = cam.generate_heatmap(dummy_x)

    assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    cam.remove_hooks()
