"""
Unit and Smoke Tests for DenseNet-201 Multi-Label Chest X-ray Pipeline
======================================================================
Tests:
  1. DenseNet201Chest14 architecture shapes (1920 -> 512 -> 256 -> 14).
  2. Multi-label logits shape (N, 14) and feature shape (N, 1920).
  3. Probability outputs in [0, 1] via Sigmoid.
  4. Memory-conscious stage 1 (freeze backbone) and stage 2 (unfreeze denseblock4).
  5. Grad-CAM targeting features.denseblock4.
  6. Multi-label evaluation and optimal Youden threshold calculation.
  7. CUDA forward, backward, and optimizer step smoke test with AMP mixed precision.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

# Path bootstrap
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from src.densenet_model import (
    CHESTMNIST_14_LABELS,
    NUM_CHEST_CLASSES,
    DenseNet201Chest14,
    build_densenet201,
)
from src.gradcam import GradCAM
from src.evaluate_densenet import (
    compute_ece,
    evaluate_predictions,
    find_optimal_threshold,
)


def test_densenet_architecture_shapes():
    """Verifies DenseNet201Chest14 output dimensions and MLP head structure."""
    model = build_densenet201(pretrained=False, dropout=0.3, activation="relu")
    model.eval()

    # Input: batch of 2 RGB images (2, 3, 224, 224)
    x = torch.randn(2, 3, 224, 224)

    # 1. Forward logits
    logits = model(x)
    assert logits.shape == (2, NUM_CHEST_CLASSES), f"Expected shape (2, 14), got {logits.shape}"
    assert NUM_CHEST_CLASSES == 14, "Must have exactly 14 classes"

    # 2. Features
    feats = model.extract_features(x)
    assert feats.shape == (2, 1920), f"Expected 1920 features, got {feats.shape}"

    # 3. Probabilities in [0, 1]
    probs = model.predict_proba(x)
    assert probs.shape == (2, 14)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    # 4. Forward with features
    l, f = model.forward_with_features(x)
    assert l.shape == (2, 14) and f.shape == (2, 1920)


def test_densenet_two_stage_freezing():
    """Tests Stage 1 freeze and Stage 2 denseblock4 unfreeze logic."""
    model = build_densenet201(pretrained=False)

    # Stage 1: Freeze backbone
    model.freeze_backbone(True)
    for p in model.features.parameters():
        assert not p.requires_grad, "Backbone parameters must be frozen in Stage 1"
    for p in model.classifier.parameters():
        assert p.requires_grad, "Classifier MLP head parameters must be trainable in Stage 1"

    # Stage 2: Unfreeze denseblock4 & norm5
    model.unfreeze_denseblock4()
    for p in model.features.denseblock1.parameters():
        assert not p.requires_grad, "denseblock1 must remain frozen in Stage 2"
    for p in model.features.denseblock2.parameters():
        assert not p.requires_grad, "denseblock2 must remain frozen in Stage 2"
    for p in model.features.denseblock3.parameters():
        assert not p.requires_grad, "denseblock3 must remain frozen in Stage 2"
    for p in model.features.denseblock4.parameters():
        assert p.requires_grad, "denseblock4 must be trainable in Stage 2"
    for p in model.features.norm5.parameters():
        assert p.requires_grad, "norm5 must be trainable in Stage 2"


def test_gradcam_densenet():
    """Tests Grad-CAM execution targeting DenseNet201 features.denseblock4."""
    model = build_densenet201(pretrained=False)
    model.eval()

    cam_gen = GradCAM(model)
    assert cam_gen.target_layer == model.features.denseblock4

    x = torch.randn(1, 3, 224, 224)
    cam = cam_gen.compute_cam(x, target_class=6)  # class 6 = pneumonia
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert not np.isnan(cam).any(), "Grad-CAM map must not contain NaNs"


def test_multilabel_evaluation_and_thresholds():
    """Tests multi-label evaluation, ECE, and optimal threshold selection."""
    rng = np.random.RandomState(42)
    n_samples = 200
    # Synthetic ground truth
    targets = (rng.rand(n_samples, 14) > 0.85).astype(int)
    # Synthetic model predictions with signal
    probs = np.clip(targets * 0.7 + rng.rand(n_samples, 14) * 0.3, 0.0, 1.0)

    results = evaluate_predictions(probs, targets, tune_thresholds=True)
    assert "macro" in results and "per_class" in results and "optimal_thresholds" in results
    assert len(results["optimal_thresholds"]) == 14

    macro = results["macro"]
    assert 0.0 <= macro["macro_auroc"] <= 1.0
    assert 0.0 <= macro["macro_f1"] <= 1.0
    assert 0.0 <= macro["macro_ece"] <= 1.0

    # Test ECE calculation directly
    ece_val = compute_ece(probs[:, 0], targets[:, 0])
    assert 0.0 <= ece_val <= 1.0

    # Test Youden threshold
    t_opt, t_m = find_optimal_threshold(probs[:, 0], targets[:, 0], method="youden")
    assert 0.05 <= t_opt <= 0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward_backward_amp_smoke():
    """
    Runs forward, backward, and optimizer steps on CUDA with AMP mixed precision.
    Verifies that batch size 4 runs without CUDA OOM on 4 GB GPU.
    """
    device = torch.device("cuda")
    model = build_densenet201(pretrained=False, device=device)
    model.train()
    model.freeze_backbone(True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.BCEWithLogitsLoss()

    batch_x = torch.randn(4, 3, 224, 224, device=device)
    batch_y = torch.randint(0, 2, (4, 14), device=device).float()

    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert not torch.isnan(loss), "Loss must not be NaN"
    assert logits.shape == (4, 14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
