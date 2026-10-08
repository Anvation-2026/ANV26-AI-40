import numpy as np
import pytest

from src.calibration import (
    compute_ece,
    fit_abstention_thresholds,
    fit_temperature,
    temperature_scale_logits,
)


def test_ece_perfect_calibration():
    # If predicted confidence exactly equals empirical accuracy in every bin, ECE ≈ 0
    n = 1000
    probs = np.full((n, 2), 0.5)
    # 50% ones, 50% zeros
    labels = np.array([1 if i % 2 == 0 else 0 for i in range(n)])
    ece = compute_ece(probs, labels, n_bins=10)
    assert ece < 0.05


def test_temperature_scaling_invariance_at_t1():
    logits = np.array([[2.0, -1.0], [0.5, 1.5], [-2.0, 3.0]])
    p_t1 = temperature_scale_logits(logits, temperature=1.0)
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    expected = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    assert np.allclose(p_t1, expected, atol=1e-6)


def test_fit_temperature_produces_positive_t():
    logits = np.random.randn(200, 2)
    labels = np.random.randint(0, 2, 200)
    t, meta = fit_temperature(logits, labels)
    assert t > 0.0
    assert meta["temperature"] == round(t, 6)


def test_fit_abstention_thresholds_behavior():
    # 1. Normal case where constraints can be satisfied
    confs = np.linspace(0.5, 0.99, 100)
    labels = np.ones(100, dtype=int)
    preds = np.ones(100, dtype=int)  # 100% accuracy
    th = fit_abstention_thresholds(confs, labels, preds, target_acc=0.95, min_coverage=0.60)
    assert 0.5 <= th["tau_accept"] <= 1.0
    assert not th["tau_accept_fallback"]

    # 2. Fallback case where impossible target accuracy is requested
    labels_err = np.array([0, 1] * 50)  # 50% accuracy always
    th_fb = fit_abstention_thresholds(confs, labels_err, preds, target_acc=0.99, min_coverage=0.90)
    assert th_fb["tau_accept_fallback"] is True
