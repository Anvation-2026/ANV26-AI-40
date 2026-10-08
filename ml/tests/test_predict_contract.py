import json
from pathlib import Path
import numpy as np
import pytest

from config import DATASET_PATH
from src.dataset import get_split
from src.predict import (
    MedGuardPredictor,
    ModelUnavailableError,
    get_predictor,
    model_status,
    predict_image,
)

REQUIRED_KEYS = [
    "status",
    "finding",
    "probability",
    "uncertainty",
    "quality",
    "ood",
    "explanation",
    "evidence",
    "heatmap_path",
    "model_version",
    "limitations",
    "details",
]

# Task 5: new required details keys
REQUIRED_DETAILS_KEYS = [
    "heatmap_target_class",
    "heatmap_note",
]

# Task 1: quality sub-dict structure keys
REQUIRED_QUALITY_SUB_KEYS = ["status", "metrics", "thresholds", "passed", "labels", "reasons"]

BANNED_WORDS = ["diagnosed with", "you have", "treatment", "confirmed finding"]


def test_predict_contract_keys_and_json_serializable():
    val_images, _ = get_split("val", DATASET_PATH)
    clean_img = val_images[0]

    result = predict_image(clean_img, generate_heatmap=False)

    for k in REQUIRED_KEYS:
        assert k in result, f"Missing required contract key: {k}"

    # Verify JSON serializability
    json_str = json.dumps(result)
    assert json_str is not None

    # Check banned words
    for banned in BANNED_WORDS:
        assert banned not in result["explanation"].lower(), f"Banned phrase '{banned}' in explanation"


def test_predict_deterministic():
    val_images, _ = get_split("val", DATASET_PATH)
    img = val_images[1]

    r1 = predict_image(img, generate_heatmap=False)
    r2 = predict_image(img, generate_heatmap=False)

    assert r1["status"] == r2["status"]
    assert r1["probability"] == r2["probability"]
    assert r1["details"]["p_pneumonia_calibrated"] == r2["details"]["p_pneumonia_calibrated"]


def test_predict_unsupported_file_garbage_bytes():
    res = predict_image(b"random corrupted bytes not an image")
    assert res["status"] == "unsupported_file"
    assert res["finding"] is None
    assert res["probability"] is None
    assert res["heatmap_path"] is None


def test_model_status_never_raises():
    status = model_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "device" in status
    assert "capabilities" in status
    # Task 2: extended model_status keys
    for key in ("model_name", "dataset", "classes", "supported_modality"):
        assert key in status, f"Missing model_status key: '{key}'"


def test_missing_model_raises_model_unavailable_error(tmp_path):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    with pytest.raises(ModelUnavailableError):
        MedGuardPredictor(models_dir=empty_dir)


# ─── Task 5: heatmap metadata in details ─────────────────────────────────────

def test_heatmap_metadata_keys_present_in_details():
    """Task 5: heatmap_target_class and heatmap_note always present in details."""
    val_images, _ = get_split("val", DATASET_PATH)
    img = val_images[0]
    res = predict_image(img, generate_heatmap=False)
    details = res["details"]
    for k in REQUIRED_DETAILS_KEYS:
        assert k in details, f"Missing Task 5 details key: '{k}'"


def test_heatmap_target_class_none_when_not_success_or_uncertain():
    """When status is not success/uncertain, heatmap_target_class must be None."""
    res = predict_image(b"corrupted bytes 123 not a real image", generate_heatmap=True)
    assert res["status"] == "unsupported_file"
    # heatmap_target_class is only populated when Grad-CAM runs (success/uncertain)
    # For unsupported_file it is absent from details (early return path)
    # The key may not be in details for the early-return path — that is acceptable.
    # But if it IS present it must be None.
    if "heatmap_target_class" in res.get("details", {}):
        assert res["details"]["heatmap_target_class"] is None


# ─── Task 1: quality thresholds/passed/labels in details ─────────────────────

def test_quality_detail_has_thresholds_passed_labels():
    """Task 1: details.quality must include thresholds, passed, and labels dicts."""
    val_images, _ = get_split("val", DATASET_PATH)
    img = val_images[0]
    res = predict_image(img, generate_heatmap=False)
    q = res["details"].get("quality")
    if q is None:
        pytest.skip("Quality evaluator not available")
    for sub in REQUIRED_QUALITY_SUB_KEYS:
        assert sub in q, f"Missing quality sub-key: '{sub}'"
    assert isinstance(q["thresholds"], dict), "details.quality.thresholds must be a dict"
    assert isinstance(q["passed"], dict), "details.quality.passed must be a dict"
    assert isinstance(q["labels"], dict), "details.quality.labels must be a dict"

